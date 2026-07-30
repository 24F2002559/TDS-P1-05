#!/usr/bin/env python3
"""
Data-Analyst Telegram Bot – FREE tier only.
Primary: AiPipe → OpenRouter → Hugging Face.
Answers every message with a single JSON object.
Debug prints to stderr (visible in Render logs).
"""

import os, sys, json, time, threading, traceback
from io import StringIO
from datetime import datetime, timezone
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from telegram.ext import Application, MessageHandler, filters

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
BASE_URL = os.environ["BASE_URL"]

AIPIPE_API_KEY = os.environ.get("AIPIPE_API_KEY", "")
AIPIPE_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://api.aipipe.ai/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

# ------------------------------------------------------------
# Logging to stderr (visible in Render)
# ------------------------------------------------------------
def log(msg: str):
    print(msg, file=sys.stderr, flush=True)

# ------------------------------------------------------------
# Global log list (served as JSONL)
# ------------------------------------------------------------
log_lines = []

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
SYSTEM_PROMPT_TOOLS = """You are a data analyst bot. Answer ONLY with a JSON object. Use the `run_python` tool to download/compute. Never guess.
- Answer the LAST user message; earlier ones are context.
- Include "log_url": "LOG_URL_PLACEHOLDER".
- If a message is only setup, reply {"answer": "ack", "log_url": "LOG_URL_PLACEHOLDER"}.
- Output ONLY the JSON, no markdown, no prose."""

SYSTEM_PROMPT_NO_TOOLS = """You are a data analyst bot. You cannot run code, so give your best answer from your knowledge.
- Reply ONLY with a JSON object matching the last user request.
- Include "log_url": "LOG_URL_PLACEHOLDER".
- If unsure, still answer as best you can. Output ONLY the JSON."""

# ------------------------------------------------------------
# LLM call helpers
# ------------------------------------------------------------
def call_openai_compatible(base_url, api_key, model, messages, tools=None):
    url = f"{base_url}/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                             json=payload, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            msg = data["choices"][0].get("message", {})
            if tools and msg.get("tool_calls"):
                return json.dumps({"message": msg})
            return msg.get("content", "")
    except Exception as e:
        log(f"OpenAI compat error: {e}")
    return None

def call_huggingface(model, messages):
    if not HF_API_KEY:
        return None
    prompt = "\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                       for m in messages if m['role'] != 'system')
    if messages and messages[0]['role'] == 'system':
        prompt = f"System: {messages[0]['content']}\n{prompt}"
    try:
        resp = requests.post(f"https://api-inference.huggingface.co/models/{model}",
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

# ------------------------------------------------------------
# Unified LLM caller
# ------------------------------------------------------------
def call_llm(messages, tools=None):
    use_tools = tools is not None
    # AiPipe
    if AIPIPE_API_KEY:
        for model in ["aipipe-v1"]:
            res = call_openai_compatible(AIPIPE_BASE_URL, AIPIPE_API_KEY, model, messages, tools if use_tools else None)
            if res is not None:
                return res
    # OpenRouter
    if OPENROUTER_API_KEY:
        for model in ["meta-llama/llama-3.3-70b-instruct:free", "google/gemini-2.0-flash-001",
                      "mistralai/mistral-7b-instruct:free", "nousresearch/hermes-3-llama-3.1-405b:free"]:
            res = call_openai_compatible("https://openrouter.ai/api/v1", OPENROUTER_API_KEY, model, messages,
                                         tools if use_tools else None)
            if res is not None:
                return res
    # Hugging Face (only if no tools)
    if not use_tools:
        for model in ["mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta"]:
            res = call_huggingface(model, messages)
            if res is not None:
                return res
    return None

# ------------------------------------------------------------
# Agent loop
# ------------------------------------------------------------
def agent_loop(history):
    deadline = time.time() + 210
    max_calls = 10
    done = 0
    tools = [{"type": "function", "function": {"name": "run_python", "description": "Run Python code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
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
                if done >= max_calls:
                    messages.append({"role": "user", "content": "Stop tools. Answer NOW."})
                    continue
                tc = msg["tool_calls"][0]
                if tc["function"]["name"] != "run_python":
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Unknown function"})
                    continue
                code = json.loads(tc["function"]["arguments"])["code"]
                log_lines.append({"time": datetime.now(timezone.utc).isoformat(), "type": "tool_call", "code": code})
                out = run_python(code)
                log_lines.append({"time": datetime.now(timezone.utc).isoformat(), "type": "tool_output", "output": out})
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
                done += 1
                continue
            else:
                raw = msg.get("content", "")
                return raw
        return raw

# ------------------------------------------------------------
# JSON extraction
# ------------------------------------------------------------
def extract_json(text):
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

def process_llm_output(raw):
    try:
        data = extract_json(raw)
    except:
        data = {"answer": raw.strip()}
    if "answer" not in data:
        data = {"answer": data}
    data["log_url"] = f"{BASE_URL}/run.jsonl"
    return data

# ------------------------------------------------------------
# Telegram handler
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
        log_lines.append({"time": datetime.now(timezone.utc).isoformat(), "chat_id": chat_id, "question": user_text, "answer": final_json.get("answer"), "raw_llm": raw})
    except Exception as e:
        reply_text = json.dumps({"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"})
        log_lines.append({"time": datetime.now(timezone.utc).isoformat(), "chat_id": chat_id, "error": str(e), "traceback": traceback.format_exc()})
        log(f"ERROR: {traceback.format_exc()}")

    # Send reply (first via PTB, then raw HTTP fallback)
    try:
        await context.bot.send_message(chat_id=chat_id, text=reply_text)
    except Exception as send_err:
        log(f"sendMessage error: {send_err}, using raw HTTP")
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          json={"chat_id": chat_id, "text": reply_text})
        except:
            pass

    history_store[chat_id].append({"role": "assistant", "content": reply_text})
    if len(history_store[chat_id]) > 20:
        history_store[chat_id] = history_store[chat_id][-20:]

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/run.jsonl")
def run_log():
    return PlainTextResponse("\n".join(json.dumps(line) for line in log_lines), media_type="text/plain")

# ------------------------------------------------------------
# Keep-alive thread
# ------------------------------------------------------------
def keep_alive():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health")
        except:
            pass

# ------------------------------------------------------------
# Bot thread with robust error logging
# ------------------------------------------------------------
def run_bot():
    log(">>> Bot thread started, building application...")
    try:
        application = Application.builder().token(BOT_TOKEN).build()
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        log(">>> Application built, starting polling...")
        application.run_polling(drop_pending_updates=False)   # consume all queued messages
        log(">>> Polling ended (should not happen)")
    except Exception as e:
        log(f"!!! Bot thread crashed: {e}\n{traceback.format_exc()}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Start keep-alive and bot threads (non-daemon so they stay alive)
    t1 = threading.Thread(target=keep_alive, daemon=True)
    t2 = threading.Thread(target=run_bot, daemon=False)   # non-daemon prevents premature exit
    t1.start()
    t2.start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
