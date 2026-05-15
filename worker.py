#!/usr/bin/env python3
import os
import sys
import io
import json
import time
import base64
import logging
import random
from datetime import datetime
from json import JSONDecodeError

import yaml
import redis
import requests
from PIL import Image
from openai import OpenAI
from json_repair import repair_json

from prompts import build_prompt

# ---------- Config & logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config.yaml') as f:
    config = yaml.safe_load(f)

api_cfg = config['api']
gen_cfg = config['generation']
BASE_URL = f"http://{api_cfg['host']}:{api_cfg['port']}/v1"
MODEL = api_cfg['model']
TIMEOUT = api_cfg.get('timeout', 120)

worker_id = os.environ.get('HOSTNAME', 'worker-unknown')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs('/shared-output', exist_ok=True)
OUTPUT_PATH = f"/shared-output/alt-text_{worker_id}_{timestamp}.jsonl"
ERROR_PATH = f"/shared-output/errors_{worker_id}_{timestamp}.json"

# ---------- LLM client ----------
client = OpenAI(api_key="EMPTY", base_url=BASE_URL, timeout=TIMEOUT)

try:
    client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )
    logger.info('LLM connection successful')
except Exception as e:
    logger.error(f'LLM connection failed: {str(e)}')
    sys.exit(1)

# ---------- Redis ----------
QUEUE = 'archival-alt-text'
PROCESSING = 'archival-alt-text:processing'

redis_client = redis.Redis(
    host=os.environ.get('REDIS_HOST', 'redis-service'),
    port=6379, db=0,
    socket_connect_timeout=10,
    socket_timeout=30,
    )


def get_next_task():
    """Returns (parsed_task, raw_bytes) or special string flags."""
    try:

        raw = redis_client.brpoplpush(QUEUE, PROCESSING, timeout=60)
        if raw:
            return json.loads(raw.decode('utf-8')), raw
        main_len = redis_client.llen(QUEUE)
        proc_len = redis_client.llen(PROCESSING)
        logger.info(f"Queue status: main={main_len}, processing={proc_len}")
        if main_len == 0:
            return "QUEUE_EMPTY", None
        return None, None
    except redis.ConnectionError as e:
        logger.error(f"Redis connection failed: {e}")
        return "REDIS_ERROR", None

def complete_task(raw):
    try:

        redis_client.lrem(PROCESSING, 1, raw)
    except Exception as e:
        logger.warning(f"Could not complete task: {e}")

def fail_task(raw):
    try:

        redis_client.lrem(PROCESSING, 1, raw)
        redis_client.lpush(QUEUE, raw)
    except Exception as e:
        logger.warning(f"Could not fail task: {e}")

# ---------- Image fetch + encode ----------
def encode_jpeg(img):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95, optimize=True, subsampling=0)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def fetch_and_encode_image(pid, max_retries=5):
    """Fetch from Islandora, resize to stay under model file size limit."""
    url = f'https://digital.lib.ku.edu/islandora/object/{pid}/datastream/OBJ/view'
    MAX_FILE_BYTES = 3355443  # ~3.2MB
    MAX_PIXEL_DIM = 4000

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Try at native size first (capped at MAX_PIXEL_DIM)
            max_dim = MAX_PIXEL_DIM
            while max_dim >= 200:
                w, h = img.size
                scale = max_dim / max(w, h)
                if scale < 1:
                    work = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                else:
                    work = img
                encoded = encode_jpeg(work)
                size_mb = len(encoded) / (1024 * 1024)
                if len(encoded) <= MAX_FILE_BYTES:
                    logger.info(f"Image encoded: {size_mb:.2f}MB at max dim {max(work.size)}")
                    return encoded
                # Shrink and retry
                max_dim = int(max_dim * ((MAX_FILE_BYTES / len(encoded)) ** 0.5) * 0.93)

            logger.warning(f"Image still {size_mb:.2f}MB at minimum size, sending anyway")
            return encoded
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

# ---------- Metadata + LLM ----------
USEFUL_DC_FIELDS = ['dc.title', 'dc.date', 'dc.creator', 'dc.description',
                    'dc.subject', 'dc.coverage']

def clean_metadata(task):
    """Return dict of useful dc.* fields, with shortened keys, dropping empties."""
    out = {}
    for f in USEFUL_DC_FIELDS:
        v = task.get(f)
        if v:
            out[f.replace('dc.', '')] = v
    return out

def derive_item_type(task):
    rels = task.get('RELS_EXT_hasModel_uri_ms', '')
    if 'page' in rels:
        return 'page'
    return 'image'

def decode_json_response(text):
    """Strip common wrapping and parse JSON, with json-repair as fallback."""
    for token in ['```json', '```', '<think>', '</think>',
                  '<|begin_of_box|>', '<|end_of_box|>']:
        text = text.replace(token, '')
    text = text.strip()
    # The assistant prefix trick means model output may not start with '{'
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        text = text + '}'
    try:
        return json.loads(text)
    except JSONDecodeError:
        try:
            return json.loads(repair_json(text))
        except Exception as e:
            logger.warning(f"Could not parse JSON response: {e}")
            return {"error": "Badly formed JSON response", "raw": text[:500]}

def query_llm(pid, image_url, metadata, item_type, max_retries=5):
    messages = build_prompt(metadata, image_url, item_type)
    # Force JSON start
    messages.append({"role": "assistant", "content": "{"})

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=gen_cfg.get('max_tokens', 512),
                temperature=gen_cfg.get('temperature', 0.3),
                top_p=gen_cfg.get('top_p', 0.9),
            )
            msg = completion.choices[0].message.content
            if not msg:
                return {"error": "Empty model response"}
            result = decode_json_response(msg)
            result['model'] = completion.model
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 2 * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM error for {pid} (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                raise

# ---------- Output ----------
def write_result(record):
    with open(OUTPUT_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')

# ---------- Main loop ----------
logger.info(f"Worker {worker_id} starting...")
processed = 0
errors = 0
consecutive_errors = 0
MAX_CONSECUTIVE = 10
error_records = []

while True:
    task = None
    pid = None
    try:
        task, raw_task = get_next_task()
        if task == "QUEUE_EMPTY":
            logger.info("Queue empty, exiting")
            break
        if task == "REDIS_ERROR":
            sys.exit(1)
        if task is None:
            time.sleep(10)
            continue

        pid = task.get('pid')
        if not pid:
            logger.warning(f"Task missing pid, skipping: {task}")
            complete_task(raw_task)
            continue

        item_type = derive_item_type(task)
        metadata = clean_metadata(task)
        logger.info(f"Processing {pid} ({item_type}) — task #{processed + 1}")

        img_enc = fetch_and_encode_image(pid)
        image_url = f"data:image/jpeg;base64,{img_enc}"

        result = query_llm(pid, image_url, metadata, item_type)

        record = {
            'pid': pid,
            'item_type': item_type,
            'alt_text': result.get('alt_text'),
            'confidence': result.get('confidence'),
            'notes': result.get('notes'),
            'model': result.get('model'),
            'error': result.get('error'),
            'timestamp': datetime.now().isoformat(),
        }
        write_result(record)
        complete_task(raw_task)

        processed += 1
        consecutive_errors = 0
        logger.info(f"Done {pid} ({processed} total)")

    except KeyboardInterrupt:
        logger.info("Interrupted")
        break
    except Exception as e:
        errors += 1
        consecutive_errors += 1
        logger.error(f"Error processing {pid}: {e}")
        error_records.append({
            'pid': pid,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        })
        if task:
            fail_task(raw_task)
        if consecutive_errors >= MAX_CONSECUTIVE:
            logger.error("Too many consecutive errors, exiting")
            sys.exit(1)

# ---------- Cleanup ----------
if error_records:
    with open(ERROR_PATH, 'w') as f:
        json.dump(error_records, f, indent=2)

try:

    logger.info(f"Final queue: main={redis_client.llen(QUEUE)}, processing={redis_client.llen(PROCESSING)}")
except Exception:
    pass

logger.info(f"Worker {worker_id} done. Processed: {processed}, Errors: {errors}")
