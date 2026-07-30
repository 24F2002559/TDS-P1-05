#!/usr/bin/env python3
"""
Data-Analyst Telegram Bot – FREE tier only.
Uses OpenRouter free models (with tool calling) + Hugging Face Inference API as fallback.
Answers every message with a single JSON object.
"""

import os
import json
import time
import threading
import traceback
import sys
from io import StringIO
from datetime import datetime, timezone
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from telegram.ext import Application, MessageHandler, filters

# ----------------------------------------------------------------------
# Environment configuration
# ----------------------------------------------------------------------
BOT_TOKEN = os.environ["BOT_TOKEN"]
BASE_URL = os.environ["BASE_URL"]                     # your public Render URL
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

# ----------------------------------------------------------------------
# Free model lists (order matters – first successful wins)
# ----------------------------------------------------------------------
OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-001",
    "mistralai/mistral-7b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

HF_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]

# ----------------------------------------------------------------------
# Global log (served at /run.jsonl)
# ----------------------------------------------------------------------
log_lines = []

# ----------------------------------------------------------------------
# Safe Python execution sandbox
# ----------------------------------------------------------------------
def run_python(code: str) -> str:
    """Execute Python and return stdout (last 8000 chars)."""
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

# ----------------------------------------------------------------------
# System prompts
# ----------------------------------------------------------------------
SYSTEM_PROMPT_TOOLS = """You are a data analyst bot. Answer ONLY with a JSON object. Use the `run_python` tool to download/compute. Never guess.
- Answer the LAST user message; earlier ones are context.
- Include "log_url": "LOG_URL_PLACEHOLDER".
- If a message is only setup, reply {"answer": "ack", "log_url": "LOG_URL_PLACEHOLDER"}.
- Output ONLY the JSON, no markdown, no prose."""

SYSTEM_PROMPT_NO_TOOLS = """You are a data analyst bot. You cannot run code, so give your best answer from your knowledge.
- Reply ONLY with a JSON object matching the last user request.
- Include "log_url": "LOG_URL_PLACEHOLDER".
- If unsure, still answer as best you can. Output ONLY the JSON."""

# ----------------------------------------------------------------------
# Unified LLM caller with fallback chain
# ----------------------------------------------------------------------
def call_llm(messages: list, tools: Optional[list] = None) -> Optional[str]:
    """
    Try OpenRouter free models first (support tools), then HF models (no tools).
    Returns the text output of the model, or None if everything fails.
    """
    use_tools = tools is not None

    # ---- OpenRouter (OpenAI-compatible chat API) ----
    if OPENROUTER_API_KEY:
        for model in OPENROUTER_MODELS:
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                }
                if use_tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"

                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    msg = choice.get("message", {})
                    # If the model returns a tool call, pack the whole message
                    if use_tools and msg.get("tool_calls"):
                        return json.dumps({"message": msg})
                    content = msg.get("content", "")
                    return content
            except Exception:
                continue

    # ---- Hugging Face Inference API (no native tools) ----
    if HF_API_KEY and not use_tools:
        for model in HF_MODELS:
            try:
                # Convert messages to a single prompt
                prompt = "\n".join(
                    f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                    for m in messages if m['role'] != 'system'
                )
                if messages and messages[0]['role'] == 'system':
                    prompt = f"System: {messages[0]['content']}\n{prompt}"

                resp = requests.post(
                    f"https://api-inference.huggingface.co/models/{model}",
                    headers={"Authorization": f"Bearer {HF_API_KEY}"},
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 1024,
                            "temperature": 0.0,
                            "return_full_text": False,
                        },
                    },
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        return data[0].get("generated_text", "")
                    elif isinstance(data, dict):
                        return data.get("generated_text", "")
            except Exception:
                continue

    # All attempts exhausted
    return None

# ----------------------------------------------------------------------
# Agent loop (LLM + tool execution)
# ----------------------------------------------------------------------
def agent_loop(history: list) -> str:
    deadline = time.time() + 210
    max_tool_calls = 10
    tool_calls_done = 0

    tools = [{
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python code to fetch/compute. Returns stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    }]

    messages = [{"role": "system", "content": SYSTEM_PROMPT_TOOLS}] + history

    while True:
        if time.time() > deadline:
            messages[0] = {"role": "system", "content": SYSTEM_PROMPT_NO_TOOLS}
            raw = call_llm(messages, tools=None)
            if raw is None:
                raw = '{"answer": "timeout error", "log_url": "LOG_URL_PLACEHOLDER"}'
            return raw

        raw = call_llm(messages, tools)
        if raw is None:
            return '{"answer": "service unavailable", "log_url": "LOG_URL_PLACEHOLDER"}'

        # OpenRouter tool call packed as JSON string
        if raw.startswith('{"message":'):
            try:
                msg = json.loads(raw)["message"]
            except Exception:
                return raw

            if msg.get("tool_calls"):
                if tool_calls_done >= max_tool_calls:
                    messages.append({"role": "user", "content": "Stop tools. Answer NOW."})
                    continue

                tool_call = msg["tool_calls"][0]
                func_name = tool_call["function"]["name"]
                if func_name != "run_python":
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                    messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": "Unknown function"})
                    continue

                code = json.loads(tool_call["function"]["arguments"])["code"]
                log_lines.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "type": "tool_call",
                    "code": code
                })
                output = run_python(code)
                log_lines.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "type": "tool_output",
                    "output": output
                })

                messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": output})
                tool_calls_done += 1
                continue
            else:
                # No tool calls – this is the final answer
                raw = msg.get("content", "")
                return raw

        # Plain text answer (or HuggingFace fallback)
        return raw

# ----------------------------------------------------------------------
# JSON extraction and final answer shaping
# ----------------------------------------------------------------------
def extract_json(text: str) -> dict:
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise ValueError("Unbalanced")

def process_llm_output(raw_output: str) -> dict:
    try:
        data = extract_json(raw_output)
    except Exception:
        data = {"answer": raw_output.strip()}
    if "answer" not in data:
        data = {"answer": data}
    data["log_url"] = f"{BASE_URL}/run.jsonl"
    return data

# ----------------------------------------------------------------------
# Per‑chat history (for multi‑turn conversations)
# ----------------------------------------------------------------------
history_store = {}

async def handle_message(update, context):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    if chat_id not in history_store:
        history_store[chat_id] = []
    history_store[chat_id].append({"role": "user", "content": user_text})

    try:
        raw = agent_loop(history_store[chat_id])
        final_json = process_llm_output(raw)
        reply_text = json.dumps(final_json)

        log_lines.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "question": user_text,
            "answer": final_json.get("answer"),
            "raw_llm": raw
        })
    except Exception as e:
        reply_text = json.dumps({
            "answer": "internal error",
            "log_url": f"{BASE_URL}/run.jsonl"
        })
        log_lines.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "chat_id": chat_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })

    await context.bot.send_message(chat_id=chat_id, text=reply_text)
    history_store[chat_id].append({"role": "assistant", "content": reply_text})
    if len(history_store[chat_id]) > 20:
        history_store[chat_id] = history_store[chat_id][-20:]

# ----------------------------------------------------------------------
# FastAPI application (health check + public log)
# ----------------------------------------------------------------------
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/run.jsonl")
def run_log():
    content = "\n".join(json.dumps(line) for line in log_lines)
    return PlainTextResponse(content, media_type="text/plain")

# ----------------------------------------------------------------------
# Keep‑alive thread (internal self‑ping, helps but not sufficient alone)
# ----------------------------------------------------------------------
def keep_alive():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health")
        except Exception:
            pass

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def run_bot():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    threading.Thread(target=run_bot, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
