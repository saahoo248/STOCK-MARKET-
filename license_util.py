import base64
import hashlib
import hmac
import json
from datetime import datetime

SECRET_FILE = "config/license_secret.txt"
LICENSE_FILE = "config/license.json"

def load_secret():
    with open(SECRET_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def make_license(name, plan, expires):
    payload = {
        "name": name,
        "plan": plan,
        "expires": expires
    }

    raw = json.dumps(
        payload,
        separators=(",", ":")
    ).encode()

    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")

    signature = hmac.new(
        load_secret().encode(),
        encoded.encode(),
        hashlib.sha256
    ).hexdigest()

    return f"{encoded}.{signature}"

def verify_license(key):
    try:
        encoded, signature = key.split(".", 1)

        expected = hmac.new(
            load_secret().encode(),
            encoded.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return False, "Invalid license signature."

        padding = "=" * (-len(encoded) % 4)

        raw = base64.urlsafe_b64decode(
            encoded + padding
        )

        data = json.loads(raw.decode())

        expires = datetime.strptime(
            data["expires"],
            "%Y-%m-%d"
        ).date()

        if expires < datetime.now().date():
            return False, "License expired."

        return True, data

    except Exception:
        return False, "Invalid license key."

def save_license(key):
    with open(LICENSE_FILE, "w", encoding="utf-8") as f:
        json.dump({"key": key}, f, indent=2)

def get_saved_license():
    try:
        with open(LICENSE_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("key")
    except Exception:
        return None
