#!/usr/bin/env python3
"""
Data-Analyst Telegram Bot – AiPipe (skip on credit exhaustion) + free fallbacks.

- Primary: AiPipe (OpenAI + OpenRouter proxy) – skipped if credit exhausted.
- Fallbacks (free): Groq → OpenRouter direct → HuggingFace.
- Internet search via DuckDuckGo (DDGS from ddgs), Wikipedia, and generic HTTP fetches.
- Thread pool, 90 s sandbox timeout, 300 s answer budget, persistent log + GitHub sync.
- Robust JSON extraction and safe argument parsing.
"""

import base64
import contextlib
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import StringIO

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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
HF_API_KEY = os.environ.get("HF_API_KEY", "")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_FILE_PATH = os.environ.get("GITHUB_FILE_PATH", "run.jsonl")

LOG_URL = f"{BASE_URL}/run.jsonl"

MAX_AGENT_STEPS = 8
PY_TIMEOUT = 90  # slightly more time for data fetches
ANSWER_BUDGET = 300  # total seconds per question


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
    except Exception:
        pass


# ------------------------------------------------------------
# 4. Safe Python sandbox – internet, Wikipedia, parsing
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
            "DDGS": __import__("ddgs").DDGS,
            "wikipedia": __import__("wikipedia"),
            "lxml": __import__("lxml"),
            "html5lib": __import__("html5lib"),
            "xlrd": __import__("xlrd"),
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
    return text[-2000:] if text else "(no output — use print())"


# ------------------------------------------------------------
# 5. System prompt – emphasises printing the result
# ------------------------------------------------------------
SYSTEM_PROMPT_TOOLS = """You are a data analyst bot. Answer ONLY with a JSON object. Use `run_python` to fetch/compute.
- Answer the LAST user message; earlier ones are context.
- Your final output must be a single JSON object with exactly two keys: "answer" and "log_url".
- The value of "answer" must EXACTLY match the shape requested by the user.

**For WHO Global Health Observatory data (indicator WHOSIS_000001):**
Use this EXACT code pattern (replace countries as needed):
import requests, pandas as pd
url = "https://ghoapi.azureedge.net/api/WHOSIS_000001"
data = requests.get(url).json()["value"]
df = pd.DataFrame(data)
# Filter for both sexes, the 7 countries, and years 2010,2019,2021
df = df[(df["Dim1"] == "SEX_BTSX") & (df["TimeDim"].isin([2010,2019,2021])) & (df["SpatialDim"].isin(["BRA","CHN","IND","IDN","MEX","ZAF","TUR"]))]
pivot = df.pivot(index="SpatialDim", columns="TimeDim", values="NumericValue")
pivot["gain"] = pivot[2019] - pivot[2010]
pivot["loss"] = pivot[2019] - pivot[2021]
pivot["ratio"] = pivot["loss"] / pivot["gain"]
result = pivot["ratio"].idxmax()
country_map = {"BRA":"Brazil","CHN":"China","IND":"India","IDN":"Indonesia","MEX":"Mexico","ZAF":"South Africa","TUR":"Turkey"}
print({"answer":{"country":country_map[result]},"log_url":"LOG_URL_PLACEHOLDER"})

- The column for country is `SpatialDim`, for year is `TimeDim`, for value is `NumericValue`, and sex is `Dim1`.
- Always filter by `Dim1 == "SEX_BTSX"` (both sexes).
- **CRITICAL: Always use `print()` to output the final JSON object. Do NOT just write the dictionary – the code must include `print(...)`. If your first attempt doesn’t print the result, correct it immediately.**

**For other datasets (MOSPI, etc.):**
- Search with `DDGS().text("MOSPI maternal mortality rate table", max_results=3)` to find a direct CSV/Excel file.
- Use `wikipedia.page(title)` for quick summaries.

**Parsing tips:**
- Always print the first 500 characters of the raw response to understand the format.
- For CSV: try `pd.read_csv(..., on_bad_lines='skip', skiprows=...)`.
- For HTML: use `pd.read_html()` or BeautifulSoup.
- For Excel: `pd.read_excel(..., engine='openpyxl')`.

**Important:**
- You have up to 8 tool calls. Use them wisely.
- If you cannot fetch the exact dataset after multiple attempts, you may use Wikipedia or your own knowledge, but only as a last resort. Never output an error object or placeholder.
- For simple arithmetic, answer directly without tools.
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
        fixed = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
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
            # If 429 with "Usage", skip provider (credits exhausted)
            if resp.status_code == 429 and "Usage" in resp.text:
                return "SKIP_PROVIDER"
            return None
    except Exception as e:
        log(f"  -> {model} error: {e}")
        return None


def call_huggingface(model, messages):
    if not HF_API_KEY:
        return None
    prompt = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
        if m["role"] != "system"
    )
    if messages and messages[0]["role"] == "system":
        prompt = f"System: {messages[0]['content']}\n{prompt}"
    try:
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
    except Exception as e:
        log(f"  -> HF {model} error: {e}")
    return None


# ------------------------------------------------------------
# 8. Unified LLM caller – AiPipe first, then free fallbacks
# ------------------------------------------------------------
def call_llm(messages, tools=None, retry=True):
    use_tools = tools is not None
    log("Trying LLM providers...")

    # ===== 1. AiPipe (skip if credit exhausted) =====
    if AIPIPE_API_KEY:
        log("  [AiPipe]")
        skip_aipipe = False
        for model in ["gpt-4", "gpt-4o-mini", "gpt-3.5-turbo"]:
            res = call_openai_compatible(
                "https://aipipe.org/openai/v1",
                AIPIPE_API_KEY,
                model,
                messages,
                tools if use_tools else None,
            )
            if res is not None:
                if res == "SKIP_PROVIDER":
                    log(
                        "  AiPipe OpenAI proxy credit exhausted – skipping remaining AiPipe models."
                    )
                    skip_aipipe = True
                    break
                return res

        if not skip_aipipe:
            for model in [
                "openai/gpt-4.1-nano",
                "openai/gpt-4o-mini",
                "anthropic/claude-3-haiku",
            ]:
                res = call_openai_compatible(
                    "https://aipipe.org/openrouter/v1",
                    AIPIPE_API_KEY,
                    model,
                    messages,
                    tools if use_tools else None,
                )
                if res is not None:
                    if res == "SKIP_PROVIDER":
                        log("  AiPipe OpenRouter proxy credit exhausted – skipping.")
                        break
                    return res

    # ===== 2. Groq (free) – updated working models =====
    if GROQ_API_KEY:
        log("  [Groq]")
        for model in [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",  # from the list you provided
            "qwen/qwen3.6-27b",  # also available
        ]:
            res = call_openai_compatible(
                "https://api.groq.com/openai/v1",
                GROQ_API_KEY,
                model,
                messages,
                tools if use_tools else None,
            )
            if res is not None:
                return res

    # ===== 3. OpenRouter direct (free models) =====
    if OPENROUTER_API_KEY:
        log("  [OpenRouter direct]")
        for model in [
            "openrouter/free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free",
        ]:
            res = call_openai_compatible(
                "https://openrouter.ai/api/v1",
                OPENROUTER_API_KEY,
                model,
                messages,
                tools if use_tools else None,
            )
            if res is not None:
                return res

    # ===== 4. HuggingFace (text only, no tools) =====
    if not use_tools and HF_API_KEY:
        log("  [HuggingFace]")
        for model in ["mistralai/Mistral-7B-Instruct-v0.3"]:
            res = call_huggingface(model, messages)
            if res is not None:
                return res

    if retry:
        log("  All providers failed. Retrying once after 2s...")
        time.sleep(2)
        return call_llm(messages, tools=tools, retry=False)

    log("  All providers failed after retry.")
    return None


# ------------------------------------------------------------
# 9. Agent loop – improved feedback for missing print
# ------------------------------------------------------------
def agent_loop(history):
    deadline = time.time() + ANSWER_BUDGET
    done = 0
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute Python code. You can use DDGS (DuckDuckGo), wikipedia, pandas, requests, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TOOLS}] + history

    while True:
        if time.time() > deadline:
            messages[0] = {"role": "system", "content": SYSTEM_PROMPT_NO_TOOLS}
            raw = call_llm(messages, None)
            return (
                raw or '{"answer": "timeout error", "log_url": "LOG_URL_PLACEHOLDER"}'
            )

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
                    messages.append(
                        {"role": "user", "content": "Stop tools. Answer NOW."}
                    )
                    continue
                tc = msg["tool_calls"][0]
                if tc["function"]["name"] != "run_python":
                    messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [tc]}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": "Unknown function",
                        }
                    )
                    continue

                # Robust argument parsing
                args_str = tc["function"]["arguments"]
                if not args_str or not args_str.strip():
                    out = "Tool call arguments were empty. Please provide a valid JSON object with a 'code' key."
                    log("Tool arguments empty – asking model to retry.")
                else:
                    try:
                        args = safe_json_loads(args_str)
                        code = args["code"]
                    except (json.JSONDecodeError, KeyError) as e:
                        log(f"Tool arguments error: {e}")
                        out = f"Error parsing arguments: {e}. Please ensure you send valid JSON with a 'code' key."
                        push_log_line(
                            json.dumps(
                                {
                                    "time": datetime.now(timezone.utc).isoformat(),
                                    "type": "tool_parse_error",
                                    "error": str(e),
                                }
                            )
                        )
                        messages.append(
                            {"role": "assistant", "content": None, "tool_calls": [tc]}
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": tc["id"], "content": out}
                        )
                        done += 1
                        continue

                if args_str and args_str.strip():
                    push_log_line(
                        json.dumps(
                            {
                                "time": datetime.now(timezone.utc).isoformat(),
                                "type": "tool_call",
                                "code": code,
                            }
                        )
                    )
                    out = run_python(code)
                    push_log_line(
                        json.dumps(
                            {
                                "time": datetime.now(timezone.utc).isoformat(),
                                "type": "tool_output",
                                "output": out,
                            }
                        )
                    )

                    # If the output is the default message (no print), give a strong hint
                    if out == "(no output — use print())":
                        out += "\n[System: Your code did not print anything. Please add `print(...)` to output the result and run the tool again.]"
                    elif out.startswith("Error:"):
                        out += "\n[System: Tool failed. Try a different approach or search for an alternative data source.]"
                else:
                    pass

                messages.append(
                    {"role": "assistant", "content": None, "tool_calls": [tc]}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": out}
                )
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
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
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
# 11. Conversation history (thread‑safe)
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

        push_log_line(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "chat_id": chat_id,
                    "question": user_text,
                    "answer": final_json.get("answer"),
                    "raw_llm": raw,
                }
            )
        )
    except Exception as e:
        reply_text = json.dumps({"answer": "internal error", "log_url": LOG_URL})
        push_log_line(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "chat_id": chat_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        )
        log(f"ERROR: {traceback.format_exc()}")

    try:
        requests.post(
            f"{TG_API}/sendMessage", json={"chat_id": chat_id, "text": reply_text}
        )
    except Exception as send_err:
        log(f"sendMessage failed: {send_err}")

    with _hist_lock:
        history_store[chat_id].append({"role": "assistant", "content": reply_text})
        if len(history_store[chat_id]) > 20:
            history_store[chat_id] = history_store[chat_id][-20:]


# ------------------------------------------------------------
# 12. Telegram polling (thread pool)
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
                timeout=65,
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
# 13. FastAPI app – inline run.jsonl
# ------------------------------------------------------------
app = FastAPI()


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.api_route("/run.jsonl", methods=["GET", "HEAD"])
def run_log():
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
