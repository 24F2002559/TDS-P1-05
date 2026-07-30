#!/usr/bin/env python3
"""
Data-Analyst Telegram Bot – FREE multi‑provider (Gemini / Groq / Together / OpenRouter / HF).
Supports tool‑calling, answers with a single JSON object, logs to GitHub.
"""

import os, sys, json, time, threading, base64, traceback
from io import StringIO
from datetime import datetime, timezone
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from telegram.ext import Application, MessageHandler, filters

# ------------------------------------------------------------
# Configuration (all from environment)
# ------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
BASE_URL = os.environ["BASE_URL"]                        # your Render URL

# --- LLM API keys (all free) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")    # aistudio.google.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")        # console.groq.com/keys
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")# together.ai/settings/api-keys
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

# --- GitHub logging ---
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_FILE_PATH = os.environ.get("GITHUB_FILE_PATH", "run.jsonl")

# ------------------------------------------------------------
# Logging to stderr (visible in Render logs)
# ------------------------------------------------------------
def log(msg: str):
    print(msg, file=sys.stderr, flush=True)

# ------------------------------------------------------------
# Global log (local backup) + GitHub push
# ------------------------------------------------------------
local_log_lines = []

def push_log_line(line: str):
    local_log_lines.append(line)
    _push_to_github(line)

def _push_to_github(json_line: str):
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return
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
        log(f"GitHub push error: {e}")

# ------------------------------------------------------------
# Safe Python sandbox
# ------------------------------------------------------------
def run_python(code: str) -> str:
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    namespace = {
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "requests": __import__("requests"),
        "BeautifulSoup": __import__("bs4").BeautifulSoup,
        "openpyxl": __import__("openpyxl"),
        "json": __import__("json"),
    }
    try:
        exec(code, namespace)
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {e}\n{mystdout.getvalue()[-8000:]}"
    sys.stdout = old_stdout
    return mystdout.getvalue()[-8000:]

# ------------------------------------------------------------
# System prompts
# ------------------------------------------------------------
SYSTEM_PROMPT_TOOLS = """You are a data analyst bot. Answer ONLY with a JSON object. Use `run_python` to fetch/compute.
- Answer the LAST user message; earlier ones are context.
- Include "log_url": "LOG_URL_PLACEHOLDER".
- If a message is only setup, reply {"answer": "ack", "log_url": "LOG_URL_PLACEHOLDER"}.
- Output ONLY the JSON, no markdown, no prose."""

SYSTEM_PROMPT_NO_TOOLS = """You are a data analyst bot. You cannot run code. Give your best answer from your knowledge.
- Reply ONLY with a JSON object matching the last user request.
- Include "log_url": "LOG_URL_PLACEHOLDER".
- Output ONLY the JSON."""

# ------------------------------------------------------------
# Multi‑provider LLM caller with extensive free fallbacks
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
                return json.dumps({"message": msg})   # tool call signal
            return msg.get("content", "")
    except Exception as e:
        log(f"Error calling {model}: {e}")
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
        log(f"HF error: {e}")
    return None

def call_llm(messages: list, tools: Optional[list] = None) -> Optional[str]:
    use_tools = tools is not None

    # 1. Gemini (Google AI Studio)
    if GEMINI_API_KEY:
        for model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]:
            res = call_openai_compatible(
                "https://generativelanguage.googleapis.com/v1beta/openai",
                GEMINI_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 2. Groq (free tier)
    if GROQ_API_KEY:
        for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"]:
            res = call_openai_compatible("https://api.groq.com/openai/v1", GROQ_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 3. Together AI (free credits, OpenAI‑compatible)
    if TOGETHER_API_KEY:
        for model in ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "mistralai/Mistral-7B-Instruct-v0.1"]:
            res = call_openai_compatible("https://api.together.xyz/v1", TOGETHER_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 4. OpenRouter (free models)
    if OPENROUTER_API_KEY:
        for model in ["meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-flash-001", "mistralai/mistral-7b-instruct:free"]:
            res = call_openai_compatible("https://openrouter.ai/api/v1", OPENROUTER_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res

    # 5. Hugging Face (text only, no tools)
    if not use_tools:
        for model in ["mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta"]:
            res = call_huggingface(model, messages)
            if res is not None:
                return res

    return None

# ------------------------------------------------------------
# Agent loop (LLM + tool execution)
# ------------------------------------------------------------
def agent_loop(history: list) -> str:
    deadline = time.time() + 210
    max_tool_calls = 10
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
                if done >= max_tool_calls:
                    messages.append({"role": "user", "content": "Stop tools. Answer NOW."})
                    continue
                tc = msg["tool_calls"][0]
                if tc["function"]["name"] != "run_python":
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Unknown function"})
                    continue
                code = json.loads(tc["function"]["arguments"])["code"]
                push_log_line(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "type": "tool_call", "code": code}))
                out = run_python(code)
                push_log_line(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "type": "tool_output", "output": out}))
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
                done += 1
                continue
            else:
                raw = msg.get("content", "")
                return raw
        return raw

# ------------------------------------------------------------
# JSON extraction and answer shaping
# ------------------------------------------------------------
def extract_json(text: str) -> dict:
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise ValueError("Unbalanced")

def process_llm_output(raw: str) -> dict:
    try:
        data = extract_json(raw)
    except:
        data = {"answer": raw.strip()}
    if "answer" not in data:
        data = {"answer": data}
    data["log_url"] = f"{BASE_URL}/run.jsonl"
    return data

# ------------------------------------------------------------
# Per‑chat history and Telegram handler
# ------------------------------------------------------------
history_store = {}

async def handle_message(update, context):
    chat_id = update.effective_chat.id
    user_text = update.message.text
    log(f"MSG from {chat_id}: {user_text}")

    if chat_id not in history_store:
        history_store[chat_id] = []
    history_store[chat_id].append({"role": "user", "content": user_text})

    try:
        raw = agent_loop(history_store[chat_id])
        final_json = process_llm_output(raw)
        reply_text = json.dumps(final_json)

        push_log_line(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "question": user_text,
            "answer": final_json.get("answer"),
            "raw_llm": raw,
        }))
    except Exception as e:
        reply_text = json.dumps({"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"})
        push_log_line(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }))
        log(f"ERROR: {traceback.format_exc()}")

    # Send reply (library first, raw HTTP fallback)
    try:
        await context.bot.send_message(chat_id=chat_id, text=reply_text)
    except Exception as send_err:
        log(f"sendMessage error: {send_err}, using raw HTTP")
        try:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": reply_text}
            )
        except:
            pass

    history_store[chat_id].append({"role": "assistant", "content": reply_text})
    if len(history_store[chat_id]) > 20:
        history_store[chat_id] = history_store[chat_id][-20:]

# ------------------------------------------------------------
# FastAPI app (health fixed for HEAD requests too)
# ------------------------------------------------------------
app = FastAPI()

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/run.jsonl")
def run_log():
    content = "\n".join(local_log_lines)
    return PlainTextResponse(content, media_type="text/plain")

# ------------------------------------------------------------
# Background threads (start at module level)
# ------------------------------------------------------------
def keep_alive():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health")
        except:
            pass

def run_bot():
    log(">>> Bot thread started, building application...")
    try:
        application = Application.builder().token(BOT_TOKEN).build()
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        log(">>> Application built, starting polling...")
        application.run_polling(drop_pending_updates=False)
        log(">>> Polling ended (should not happen)")
    except Exception as e:
        log(f"!!! Bot thread crashed: {e}\n{traceback.format_exc()}")

threading.Thread(target=keep_alive, daemon=True).start()
threading.Thread(target=run_bot, daemon=False).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
