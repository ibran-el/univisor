import os
import json
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Load the Base64-encoded credentials
credentials_base64 = os.environ.get("GOOGLE_CRED_BASE64")

if not credentials_base64:
    raise ValueError("Environment variable GOOGLE_CREDENTIALS_BASE64 is not set")

# Decode the Base64 string
credentials_json = base64.b64decode(credentials_base64).decode("utf-8")
credentials_dict = json.loads(credentials_json)

# Authenticate with Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials_dict, scope)

# Use the credentials (example: access a Google Sheet)
# client = gspread.authorize(creds)
# sheet = client.open("univisor logs").sheet1
# sheet.append_row(["CHAT ID","SESSION ID", "TIMESTAMP", "TG_USERNAME", "SENDER", "MESSAGE"])

print("Successfully authenticated with Google Sheets API")