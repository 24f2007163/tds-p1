import json
import time
import os
import threading

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, PlainTextResponse

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters
)
from dotenv import load_dotenv


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL",
    "https://api.groq.com/openai/v1"
)
LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "openai/gpt-oss-120b"
)
LOG_URL = os.environ["LOG_URL"]

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

LOG_FILE = "run.jsonl"


# --------------------------------------------------
# Public web server
# --------------------------------------------------

web_app = FastAPI()


@web_app.get("/")
def root():
    return {
        "service": "data-analyst-telegram-bot",
        "log_url": LOG_URL
    }


@web_app.head("/")
def root_head():
    return Response(status_code=200)


@web_app.get("/health")
def health():
    return {
        "ok": True,
        "log_url": LOG_URL
    }


@web_app.head("/health")
def health_head():
    return Response(status_code=200)


@web_app.get("/run.jsonl")
def get_run_log():
    if os.path.exists(LOG_FILE):
        return FileResponse(
            path=LOG_FILE,
            media_type="application/x-ndjson",
            filename="run.jsonl"
        )

    return PlainTextResponse(
        "",
        media_type="application/x-ndjson"
    )


def run_web_server():
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        web_app,
        host="0.0.0.0",
        port=port
    )


# --------------------------------------------------
# Logging
# --------------------------------------------------

conversation_history = {}


def log_event(event: dict):
    event["timestamp"] = time.time()

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            json.dumps(event, ensure_ascii=False) + "\n"
        )


# --------------------------------------------------
# Telegram message handler
# --------------------------------------------------

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    log_event({
        "type": "incoming",
        "chat_id": chat_id,
        "text": user_text
    })

    history = conversation_history.setdefault(chat_id, [])

    history.append({
        "role": "user",
        "content": user_text
    })

    # Keep only the latest messages so history does not grow forever.
    del history[:-20]

    system_prompt = (
        "You are a careful data analyst. "
        "The user's LAST message asks a data-analysis question and tells you "
        "exactly what JSON shape to reply with. Earlier messages are context. "
        "Work out the real answer using the data in the conversation, public "
        "statistics you reliably know, or arithmetic on the supplied values. "
        "Reply with ONLY the exact JSON object requested by the user. "
        "Do not include explanations, markdown, or code fences. "
        "Do not add extra keys inside answer."
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                }
            ] + history[-6:]
        )

        reply_text = (
            response.choices[0]
            .message
            .content
            .strip()
        )

        history.append({
            "role": "assistant",
            "content": reply_text
        })

        try:
            parsed = json.loads(reply_text)

        except json.JSONDecodeError:
            start = reply_text.find("{")
            end = reply_text.rfind("}")

            if start == -1 or end == -1:
                raise ValueError(
                    "The model did not return a JSON object."
                )

            parsed = json.loads(
                reply_text[start:end + 1]
            )

        if not isinstance(parsed, dict):
            parsed = {
                "answer": parsed
            }

        # Always replace the model's log URL with the real one.
        parsed["log_url"] = LOG_URL

        final_reply = json.dumps(
            parsed,
            ensure_ascii=False
        )

    except Exception as error:
        log_event({
            "type": "error",
            "chat_id": chat_id,
            "error": str(error)
        })

        final_reply = json.dumps({
            "answer": "internal error",
            "log_url": LOG_URL
        })

    log_event({
        "type": "outgoing",
        "chat_id": chat_id,
        "text": final_reply
    })

    await update.message.reply_text(final_reply)


# --------------------------------------------------
# Start Telegram bot and web server
# --------------------------------------------------

telegram_app = (
    ApplicationBuilder()
    .token(TELEGRAM_BOT_TOKEN)
    .build()
)

telegram_app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message
    )
)


if __name__ == "__main__":
    
    # This also ensures run.jsonl is not empty after deployment.
    log_event({
        "type": "startup",
        "log_url": LOG_URL
    })

    web_thread = threading.Thread(
        target=run_web_server,
        daemon=True
    )

    web_thread.start()

    port = int(os.environ.get("PORT", "8000"))

    print("Bot and web server are running...")
    print(f"Web server port: {port}")
    print(f"Public log URL: {LOG_URL}")
    print("Press Ctrl+C to stop.")

    telegram_app.run_polling()