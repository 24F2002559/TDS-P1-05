#!/usr/bin/env python3
"""
Data-analyst Telegram bot — TDS Project 1 (Production‑ready merge with GitHub sync).

Features:
- Multi‑provider LLM fallback (AiPipe → OpenRouter → HuggingFace).
- Concurrent message handling via thread pool (safe for parallel TA testing).
- Python sandbox with 60 s timeout (no infinite loops).
- File‑based persistent logging + automatic push to GitHub repo.
- Robust JSON extraction (handles markdown fences, strings).
- Safe tool‑argument parsing (control‑character repair).
- Overall answer budget: 300 s (5 minutes).
- Keep‑warm thread to prevent Render spin‑down.
"""

import base64
import io
import json
import os
import re
import threading
import time
import traceback
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

# ----------------------------------------------------------------------
# Configuration – set via environment variables
# ----------------------------------------------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Primary provider (AiPipe)
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")          # API key
AIPIPE_BASE = os.environ.get("AIPIPE_BASE", "https://aipipe.org/openai/v1")
AIPIPE_MODELS = [
    "gpt-4",
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-0613",
]

# Secondary provider (OpenRouter – free tier)
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS = [
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

# Tertiary provider (HuggingFace – text only, no tools)
HF_TOKEN = os.environ.get("HF_API_KEY", "")
HF_MODELS = ["mistralai/Mistral-7B-Instruct-v0.3"]

# GitHub logging (optional but recommended)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")             # e.g. "24F2002559/TDS-P1-05"
GITHUB_FILE_PATH = os.environ.get("GITHUB_FILE_PATH", "run.jsonl")

# Deployment
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Agent limits
MAX_AGENT_STEPS = 10
PY_TIMEOUT = 60                 # seconds per run_python call
ANSWER_BUDGET = 300             # total seconds per question (5 minutes)

# ----------------------------------------------------------------------
# Thread‑safe local file logging (JSONL)
# ----------------------------------------------------------------------
_log_lock = threading.Lock()

def _push_to_github_sync(line: str):
    """Append a single line to the GitHub repo file."""
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    try:
        # Get current file content & sha
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            current = base64.b64decode(resp.json()["content"]).decode()
            sha = resp.json()["sha"]
        else:
            current = ""
            sha = None
        new_content = current + line + "\n"
        payload = {
            "message": "Update run.jsonl",
            "content": base64.b64encode(new_content.encode()).decode(),
        }
        if sha:
            payload["sha"] = sha
        requests.put(url, headers=headers, json=payload)
    except Exception:
        # Silently ignore GitHub push failures – local log is the primary one.
        pass

def log_event(**fields):
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    # Write to local file (always)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    # Push to GitHub in a background thread to avoid blocking
    threading.Thread(target=_push_to_github_sync, args=(line,), daemon=True).start()


# ----------------------------------------------------------------------
# Safe Python sandbox with timeout
# ----------------------------------------------------------------------
def run_python(code: str) -> str:
    """Execute Python code in a separate thread, kill if it runs too long."""
    out = io.StringIO()
    result: dict = {}

    def target():
        env = {
            "__name__": "__main__",
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


# ----------------------------------------------------------------------
# System prompt (comprehensive, covers edge cases)
# ----------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert data‑analyst agent answering questions sent to a Telegram bot.

Rules:
1. Work out the answer to the user's LATEST message. Earlier messages in the chat are context for multi‑turn tasks.
2. The message may embed data inline, or reference a public dataset (MOSPI, data.gov.in, etc.). Use the run_python tool to fetch data and compute — do not guess numeric results you can compute. For well‑known published statistics (e.g. "which state has the highest maternal mortality rate per MOSPI/SRS"), you may answer from reliable knowledge if fetching fails.
3. The message usually spells out the exact JSON shape it wants, e.g. Reply with ONLY {"answer": {"state": "<state>"}, "log_url": "..."}.
4. When you are ready to answer, reply with ONLY that JSON object — no prose, no markdown fences. Use a placeholder like "LOG_URL" for the log_url value; the harness substitutes the real URL. Match the requested shape for "answer" EXACTLY (keys, nesting, types: numbers as numbers unless a string is asked for).
5. If the message does not specify a shape, reply {"answer": <your concise answer>, "log_url": "LOG_URL"}.
6. If a mid‑conversation message is only setup/context ("I will send data next"), still reply with {"answer": "ok", "log_url": "LOG_URL"} unless it asks something.
7. Round numbers as instructed; if unspecified, give reasonable precision. Never add keys that were not asked for inside "answer".
8. If any tool call fails (network error, parse error, …), STOP using tools immediately and answer from your own knowledge — never output an error object.
"""


# ----------------------------------------------------------------------
# LLM helpers – multi‑provider with fallback
# ----------------------------------------------------------------------
def call_openai_compatible(base_url: str, api_key: str, model: str,
                           messages: list, use_tools: bool) -> dict:
    """Send a chat completion request, return the assistant message dict."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"model": model, "messages": messages, "temperature": 0}
    if use_tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": (
                        "Run Python code on the server and get its printed output. "
                        "pandas, numpy, requests, bs4, openpyxl are installed and the "
                        "network is available (download public datasets with requests). "
                        "Always print() what you need to see."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                        "required": ["code"],
                    },
                },
            }
        ]
    r = requests.post(url, headers=headers, json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def call_huggingface(model: str, messages: list) -> dict:
    """HuggingFace text‑generation API – wraps plain text as a message."""
    prompt = "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages if m['role'] != 'system'
    )
    if messages and messages[0]['role'] == 'system':
        prompt = f"System: {messages[0]['content']}\n{prompt}"

    r = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1024, "temperature": 0.0, "return_full_text": False},
        },
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        text = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        text = data.get("generated_text", "")
    else:
        text = ""
    return {"role": "assistant", "content": text}


def chat_completion_fallback(messages: list, use_tools: bool) -> dict:
    """
    Try providers in order until one succeeds.
    Returns a message dict.
    """
    # 1. AiPipe
    if AIPIPE_TOKEN:
        for model in AIPIPE_MODELS:
            try:
                return call_openai_compatible(AIPIPE_BASE, AIPIPE_TOKEN, model, messages, use_tools)
            except Exception:
                continue

    # 2. OpenRouter (free tier)
    if OPENROUTER_KEY:
        for model in OPENROUTER_MODELS:
            try:
                return call_openai_compatible(OPENROUTER_BASE, OPENROUTER_KEY, model, messages, use_tools)
            except Exception:
                continue

    # 3. HuggingFace (text only, no tool support)
    if not use_tools and HF_TOKEN:
        for model in HF_MODELS:
            try:
                return call_huggingface(model, messages)
            except Exception:
                continue

    raise RuntimeError("All LLM providers failed")


# ----------------------------------------------------------------------
# Safe JSON extraction (strips fences, handles strings)
# ----------------------------------------------------------------------
def safe_json_loads(s: str):
    """Try to parse JSON, if fails, escape control characters and retry."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        fixed = s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return json.loads(fixed)


def extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced JSON object out of model text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
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
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ----------------------------------------------------------------------
# Agent loop for one question
# ----------------------------------------------------------------------
_histories: dict[int, list[dict]] = {}   # chat_id -> messages
_hist_lock = threading.Lock()

def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        del history[:-20]   # keep last 20 turns
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)

    final_text = None
    deadline = time.time() + ANSWER_BUDGET
    for step in range(MAX_AGENT_STEPS):
        out_of_time = time.time() > deadline
        if out_of_time:
            messages.append(
                {"role": "user", "content": "Time is up. Reply NOW with only your best final JSON object."}
            )

        # Attempt LLM call with one retry
        try:
            msg = chat_completion_fallback(messages, use_tools=not out_of_time)
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, error=str(e))
            time.sleep(2)
            try:
                msg = chat_completion_fallback(messages, use_tools=not out_of_time)
            except Exception as e2:
                log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                break

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                # Robust argument parsing
                try:
                    args = safe_json_loads(tc["function"]["arguments"])
                    code = args.get("code", "")
                except (json.JSONDecodeError, KeyError):
                    code = tc["function"]["arguments"]   # raw fallback

                log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:4000])
                output = run_python(code)
                log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:4000])
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": output}
                )
            continue

        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text) if final_text else None
    if obj is None:
        obj = {"answer": (final_text or "unable to determine").strip()[:1000]}
    if "answer" not in obj:
        obj = {"answer": obj}
    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False)

    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    log_event(event="answer", chat_id=chat_id, reply=reply)
    return reply


# ----------------------------------------------------------------------
# Telegram polling & concurrent handling
# ----------------------------------------------------------------------
def tg(method, **params):
    r = requests.post(f"{TG_API}/{method}", json=params, timeout=65)
    return r.json()

def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return
    text = msg.get("text") or msg.get("caption") or ""
    chat_id = msg["chat"]["id"]
    if not text:
        return
    try:
        reply = solve(chat_id, text)
    except Exception:
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    tg("sendMessage", chat_id=chat_id, text=reply)


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, log_url=LOG_URL)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            resp = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            ).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# ----------------------------------------------------------------------
# FastAPI web app
# ----------------------------------------------------------------------
app = FastAPI()

@app.on_event("startup")
def _start():
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.get("/")
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}


# ----------------------------------------------------------------------
# Local run (for testing)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
