#!/usr/bin/env python3
"""
Data-Analyst Telegram Bot – production version (all free providers).

- In‑memory logging served as plain text + GitHub sync.
- Multi‑provider fallback: AiPipe → Groq → Together → OpenRouter → HuggingFace.
- Thread pool for concurrent TA testing.
- Python sandbox timeout (60 s), 300 s answer budget, LLM retries.
- Robust JSON extraction & safe argument parsing.
"""

import base64
import json
import os
import re
import sys               # ← FIX: missing import caused NameError
import threading
import time
import traceback
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import StringIO
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
BASE_URL = os.environ["BASE_URL"]
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

AIPIPE_API_KEY = os.environ.get("AIPIPE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")          # new
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")  # new
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_FILE_PATH = os.environ.get("GITHUB_FILE_PATH", "run.jsonl")

LOG_URL = f"{BASE_URL}/run.jsonl"

MAX_AGENT_STEPS = 10
PY_TIMEOUT = 60             # seconds per run_python call
ANSWER_BUDGET = 300         # total seconds per question (5 min)

# ------------------------------------------------------------
# 2. Logging helper (stderr)
# ------------------------------------------------------------
def log(msg: str):
    print(msg, file=sys.stderr, flush=True)

# ------------------------------------------------------------
# 3. In‑memory log + GitHub sync (non‑blocking)
# ------------------------------------------------------------
local_log_lines = []

def push_log_line(line: str):
    """Append to in‑memory list and push to GitHub in a background thread."""
    local_log_lines.append(line)
    # GitHub sync
    if GITHUB_TOKEN and GITHUB_REPO:
        threading.Thread(target=_push_to_github, args=(line,), daemon=True).start()

def _push_to_github(json_line: str):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            current = base64.b64decode(resp.json()["content"]).decode()
            sha = resp.json()["sha"]
        else:
            current = ""
            sha = None
        new_content = current + json_line + "\n"
        payload = {
            "message": "Update run.jsonl",
            "content": base64.b64encode(new_content.encode()).decode(),
        }
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except Exception as e:
        # Silently ignore GitHub push failures – the endpoint still works.
        pass

# ------------------------------------------------------------
# 4. Safe Python sandbox WITH TIMEOUT
# ------------------------------------------------------------
def run_python(code: str) -> str:
    out = StringIO()
    result = {}

    def target():
        env = {
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "requests": __import__("requests"),
            "BeautifulSoup": __import__("bs4").BeautifulSoup,
            "openpyxl": __import__("openpyxl"),
            "json": __import__("json"),
        }
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, env)
            result["ok"] = True
        except Exception:
            result["ok"] = False
            out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        return f"ERROR: code timed out after {PY_TIMEOUT}s"
    text = out.getvalue()
    return text[-8000:] if text else "(no output — use print())"

# ------------------------------------------------------------
# 5. System prompts (your exact wording)
# ------------------------------------------------------------
SYSTEM_PROMPT_TOOLS = """You are a data analyst bot. Answer ONLY with a JSON object. Use `run_python` to fetch/compute.
- Answer the LAST user message; earlier ones are context.
- Your final output must be a single JSON object with exactly two keys: "answer" and "log_url".
- The value of "answer" must EXACTLY match the shape requested by the user. For example, if the user asks for {"state": "<state name>"}, your output must be {"answer": {"state": "Assam"}, "log_url": "LOG_URL_PLACEHOLDER"}.
- If any tool call fails for any reason (parse error, network error, missing data, etc.), STOP using tools immediately and use your own internal knowledge to provide the best possible answer in the exact requested shape. Never output an error object, error message, or "error" key.
- Never output markdown, prose, or extra text. Only the JSON."""

SYSTEM_PROMPT_NO_TOOLS = """You are a data analyst bot. You cannot run code. Give your best answer from your knowledge.
- Reply ONLY with a JSON object matching the last user request.
- Include "log_url": "LOG_URL_PLACEHOLDER".
- The answer must be in the exact JSON shape requested. Never output an error object.
- Output ONLY the JSON."""

# ------------------------------------------------------------
# 6. Safe JSON loader (fixes control characters)
# ------------------------------------------------------------
def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        fixed = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return json.loads(fixed)

# ------------------------------------------------------------
# 7. LLM helpers – all providers
# ------------------------------------------------------------
def call_openai_compatible(base_url, api_key, model, messages, tools=None):
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": 0}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            msg = data["choices"][0].get("message", {})
            if tools and msg.get("tool_calls"):
                return json.dumps({"message": msg})
            return msg.get("content", "")
        else:
            log(f"  -> {model} returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log(f"  -> {model} error: {e}")
    return None

def call_huggingface(model, messages):
    if not HF_API_KEY:
        return None
    prompt = "\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                       for m in messages if m['role'] != 'system')
    if messages and messages[0]['role'] == 'system':
        prompt = f"System: {messages[0]['content']}\n{prompt}"
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 1024, "temperature": 0.0, "return_full_text": False}},
            timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "")
            elif isinstance(data, dict):
                return data.get("generated_text", "")
    except Exception as e:
        log(f"  -> HF {model} error: {e}")
    return None

# ------------------------------------------------------------
# 8. Unified LLM caller with RETRY
# ------------------------------------------------------------
def call_llm(messages, tools=None, retry=True):
    use_tools = tools is not None
    log("Trying LLM providers...")

    # 1. AiPipe
    if AIPIPE_API_KEY:
        log("  [AiPipe]")
        for model in ["gpt-4", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-0613"]:
            res = call_openai_compatible("https://aipipe.org/openai/v1", AIPIPE_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 2. Groq (free)
    if GROQ_API_KEY:
        log("  [Groq]")
        for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]:
            res = call_openai_compatible("https://api.groq.com/openai/v1", GROQ_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 3. Together AI (free)
    if TOGETHER_API_KEY:
        log("  [Together]")
        for model in ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "mistralai/Mistral-7B-Instruct-v0.1"]:
            res = call_openai_compatible("https://api.together.xyz/v1", TOGETHER_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 4. OpenRouter (free tier – updated model list)
    if OPENROUTER_API_KEY:
        log("  [OpenRouter]")
        for model in [
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemini-2.0-flash-001",          # still free on OpenRouter
            "mistralai/mistral-7b-instruct:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
        ]:
            res = call_openai_compatible("https://openrouter.ai/api/v1", OPENROUTER_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 5. HuggingFace (text only, no tools)
    if not use_tools and HF_API_KEY:
        log("  [HuggingFace]")
        for model in ["mistralai/Mistral-7B-Instruct-v0.3"]:
            res = call_huggingface(model, messages)
            if res is not None:
                return res

    # Retry once after a short delay
    if retry:
        log("  All providers failed. Retrying once after 2s...")
        time.sleep(2)
        return call_llm(messages, tools=tools, retry=False)

    log("  All providers failed after retry.")
    return None

# ------------------------------------------------------------
# 9. Agent loop (your logic, but with 300 s budget)
# ------------------------------------------------------------
def agent_loop(history):
    deadline = time.time() + ANSWER_BUDGET
    done = 0
    tools = [{
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python code to fetch/compute.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        }
    }]
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TOOLS}] + history

    while True:
        if time.time() > deadline:
            messages[0] = {"role": "system", "content": SYSTEM_PROMPT_NO_TOOLS}
            raw = call_llm(messages, None)
            return raw or '{"answer": "timeout error", "log_url": "LOG_URL_PLACEHOLDER"}'

        raw = call_llm(messages, tools)
        if raw is None:
            return '{"answer": "service unavailable", "log_url": "LOG_URL_PLACEHOLDER"}'

        if raw.startswith('{"message":'):
            try:
                msg = json.loads(raw)["message"]
            except:
                return raw
            if msg.get("tool_calls"):
                if done >= MAX_AGENT_STEPS:
                    messages.append({"role": "user", "content": "Stop tools. Answer NOW."})
                    continue
                tc = msg["tool_calls"][0]
                if tc["function"]["name"] != "run_python":
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Unknown function"})
                    continue

                # Robust argument parsing
                try:
                    args = safe_json_loads(tc["function"]["arguments"])
                    code = args["code"]
                except (json.JSONDecodeError, KeyError) as e:
                    log(f"Tool arguments error: {e}")
                    out = f"Error parsing arguments: {e}. Please answer without using tools."
                    push_log_line(json.dumps({
                        "time": datetime.now(timezone.utc).isoformat(),
                        "type": "tool_parse_error",
                        "error": str(e)
                    }))
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
                    done += 1
                    continue

                push_log_line(json.dumps({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "type": "tool_call",
                    "code": code
                }))
                out = run_python(code)
                push_log_line(json.dumps({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "type": "tool_output",
                    "output": out
                }))
                if out.startswith("Error:"):
                    out += "\n[System: Tool failed. Use your own knowledge to answer in the requested JSON shape.]"
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
                done += 1
                continue
            else:
                raw = msg.get("content", "")
                return raw
        return raw

# ------------------------------------------------------------
# 10. JSON extraction & answer shaping
# ------------------------------------------------------------
def extract_json(text):
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None

def process_llm_output(raw):
    try:
        data = extract_json(raw)
    except:
        data = {"answer": raw.strip()}
    if "answer" not in data:
        data = {"answer": data}
    data["log_url"] = LOG_URL
    return data

# ------------------------------------------------------------
# 11. Conversation history (thread‑safe, for multi‑turn)
# ------------------------------------------------------------
history_store = {}
_hist_lock = threading.Lock()

def process_message(chat_id, user_text):
    with _hist_lock:
        if chat_id not in history_store:
            history_store[chat_id] = []
        history_store[chat_id].append({"role": "user", "content": user_text})
        history = list(history_store[chat_id])

    try:
        raw = agent_loop(history)
        final_json = process_llm_output(raw)
        reply_text = json.dumps(final_json)

        push_log_line(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "question": user_text,
            "answer": final_json.get("answer"),
            "raw_llm": raw
        }))
    except Exception as e:
        reply_text = json.dumps({"answer": "internal error", "log_url": LOG_URL})
        push_log_line(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        log(f"ERROR: {traceback.format_exc()}")

    # Send reply
    try:
        requests.post(f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": reply_text})
    except Exception as send_err:
        log(f"sendMessage failed: {send_err}")

    with _hist_lock:
        history_store[chat_id].append({"role": "assistant", "content": reply_text})
        if len(history_store[chat_id]) > 20:
            history_store[chat_id] = history_store[chat_id][-20:]

# ------------------------------------------------------------
# 12. Telegram polling (with THREAD POOL)
# ------------------------------------------------------------
def telegram_polling():
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    log(">>> Telegram polling started")
    while True:
        try:
            resp = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65
            )
            if resp.status_code == 200:
                for upd in resp.json().get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message")
                    if msg and "text" in msg:
                        chat_id = msg["chat"]["id"]
                        user_text = msg["text"]
                        log(f"MSG from {chat_id}: {user_text}")
                        pool.submit(process_message, chat_id, user_text)
        except Exception as e:
            log(f"Polling error: {e}")
            time.sleep(5)

# ------------------------------------------------------------
# 13. FastAPI app – inline run.jsonl (not a file download)
# ------------------------------------------------------------
app = FastAPI()

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.api_route("/run.jsonl", methods=["GET", "HEAD"])
def run_log():
    """Return the accumulated log lines as plain text (displays inline)."""
    content = "\n".join(local_log_lines)
    return PlainTextResponse(content, media_type="text/plain")

# ------------------------------------------------------------
# 14. Keep‑alive & startup
# ------------------------------------------------------------
def keep_alive():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health")
        except:
            pass

threading.Thread(target=keep_alive, daemon=True).start()
threading.Thread(target=telegram_polling, daemon=False).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
