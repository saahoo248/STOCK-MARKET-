import os
import secrets
from license_util import make_license

SECRET_FILE = "config/license_secret.txt"

if not os.path.exists(SECRET_FILE):
    secret = secrets.token_urlsafe(48)

    with open(
        SECRET_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(secret)

print("")
print("=======================================")
print(" LICENSE GENERATOR")
print("=======================================")

name = input("License owner name: ").strip()
plan = input("Plan name (Student/Pro/Enterprise): ").strip()
expires = input("Expiry date (YYYY-MM-DD): ").strip()

key = make_license(
    name,
    plan,
    expires
)

print("")
print("LICENSE KEY:")
print(key)
print("")
print("Keep this key private.")
