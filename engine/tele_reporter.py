import json
import os
import http.client


def send_alert_to_telegram(text: str, photo_bytes: bytes = None):
    """
    Send a text message or photo+caption to Telegram via Bot API.
    Uses only stdlib (http.client) — no external dependencies.
    """
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        print("[Telegram] TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set. Skipping.")
        return False

    try:
        if not photo_bytes:
            return _send_text(token, chat_id, text)
        else:
            return _send_photo(token, chat_id, text, photo_bytes)
    except Exception as e:
        print(f"[Telegram] Unexpected error: {e}")
        return False


def _send_text(token: str, chat_id: str, text: str) -> bool:
    """Send a plain-text message."""
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }).encode("utf-8")

    conn = http.client.HTTPSConnection("api.telegram.org", timeout=15)
    try:
        conn.request(
            "POST",
            f"/bot{token}/sendMessage",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = resp.read()

        if resp.status != 200:
            print(f"[Telegram] sendMessage failed ({resp.status}): {body[:300]}")
            return False

        print("[Telegram] Text message sent.")
        return True
    finally:
        conn.close()


def _send_photo(token: str, chat_id: str, caption: str, photo_bytes: bytes) -> bool:
    """Send a photo with caption using multipart/form-data."""
    boundary = "----ChronosForecasterBoundary9x8z7"

    # Build multipart body manually
    parts = []

    # chat_id
    parts.append(f"--{boundary}\r\n"
                 f"Content-Disposition: form-data; name=\"chat_id\"\r\n\r\n"
                 f"{chat_id}\r\n")

    # parse_mode
    parts.append(f"--{boundary}\r\n"
                 f"Content-Disposition: form-data; name=\"parse_mode\"\r\n\r\n"
                 f"Markdown\r\n")

    # caption
    if caption:
        parts.append(f"--{boundary}\r\n"
                     f"Content-Disposition: form-data; name=\"caption\"\r\n\r\n"
                     f"{caption}\r\n")

    text_body = "".join(parts).encode("utf-8")

    # photo binary part
    photo_header = (f"--{boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"photo\"; filename=\"forecast.png\"\r\n"
                    f"Content-Type: image/png\r\n\r\n").encode("utf-8")

    closing = f"\r\n--{boundary}--\r\n".encode("utf-8")

    body = text_body + photo_header + photo_bytes + closing

    conn = http.client.HTTPSConnection("api.telegram.org", timeout=30)
    try:
        conn.request(
            "POST",
            f"/bot{token}/sendPhoto",
            body=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        resp_body = resp.read()

        if resp.status != 200:
            print(f"[Telegram] sendPhoto failed ({resp.status}): {resp_body[:300]}")
            return False

        print("[Telegram] Photo sent successfully.")
        return True
    finally:
        conn.close()
