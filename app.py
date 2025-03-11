from flask import Flask, jsonify, request
from deepface import DeepFace
import os
import pickle
import io
from werkzeug.utils import secure_filename
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
import json

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp"
ENCODINGS_FILE = "encodings.pkl"
CREDENTIALS_FILE = "/etc/secrets/credentials.json"
DRIVE_FOLDER_ID_KNOWN = "17zt840iXARAn9sQmDWmqpy4pdSTQLRdS"
ENCODINGS_FOLDER_ID = "1g1qZ3X6Yq6l_PQh79pVGbYCe7xwvkiyR"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure necessary folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Google Drive authentication
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Check if credentials.json is valid
if not os.path.exists(CREDENTIALS_FILE) or os.stat(CREDENTIALS_FILE).st_size == 0:
    raise FileNotFoundError("Error: credentials.json is missing or empty.")

with open(CREDENTIALS_FILE, "r") as f:
    try:
        json.load(f)  # Validate JSON format
    except json.JSONDecodeError:
        raise ValueError("Error: credentials.json contains invalid JSON.")

credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=credentials)

def download_known_faces():
    """Download all known face images from Google Drive to temp/"""
    query = f"'{DRIVE_FOLDER_ID_KNOWN}' in parents"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    for file in files:
        file_id = file["id"]
        filename = file["name"]
        file_path = os.path.join(TEMP_FOLDER, filename)

        request = drive_service.files().get_media(fileId=file_id)
        with open(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        
        print(f"✅ {filename} downloaded to temp/")

def download_encodings_file():
    """Download encodings.pkl from Google Drive."""
    query = f"'{ENCODINGS_FOLDER_ID}' in parents and name='{ENCODINGS_FILE}'"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    files = results.get("files", [])

    if files:
        file_id = files[0]["id"]
        request = drive_service.files().get_media(fileId=file_id)
        with open(ENCODINGS_FILE, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print("✅ Downloaded encodings.pkl from Google Drive.")

def upload_file_to_drive(file_path, folder_id, filename):
    """Upload a file to Google Drive."""
    file_metadata = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="application/octet-stream")

    query = f"'{folder_id}' in parents and name='{filename}'"
    existing_files = drive_service.files().list(q=query, fields="files(id)").execute().get("files", [])

    if existing_files:
        file_id = existing_files[0]["id"]
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        print(f"✅ Updated {filename} on Google Drive.")
    else:
        drive_service.files().create(body=file_metadata, media_body=media).execute()
        print(f"✅ Uploaded {filename} to Google Drive.")

# Download known faces and encodings at startup
download_known_faces()
download_encodings_file()

# Initialize known_encodings
global known_encodings
known_encodings = {}
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings = pickle.load(f)

@app.route("/update_encodings", methods=["GET"])
def update_encodings():
    """Update encodings and upload to Google Drive."""
    global known_encodings
    
    download_known_faces()
    download_encodings_file()
    known_encodings = {}
    for filename in os.listdir(TEMP_FOLDER):
        file_path = os.path.join(TEMP_FOLDER, filename)
        try:
            embeddings = DeepFace.represent(img_path=file_path, model_name="Facenet", enforce_detection=False)
            if embeddings:
                known_encodings[filename] = embeddings[0]["embedding"]
        except:
            print(f"⚠️ Unable to encode {filename}")
    
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_encodings, f)

    upload_file_to_drive(ENCODINGS_FILE, ENCODINGS_FOLDER_ID, ENCODINGS_FILE)

    return jsonify({"message": "Encodings updated, saved to Google Drive, and refreshed from Drive."})

@app.route("/match_uploaded_face", methods=["POST"])
def match_uploaded_face():
    """Match an uploaded face against known faces."""
    global known_encodings
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
        result = DeepFace.find(img_path=file_path, db_path=TEMP_FOLDER, model_name="Facenet")
        if len(result) > 0:
            matched_name = os.path.basename(result[0]['identity'].values[0])
            return jsonify({"match": True, "matched_with": matched_name})
    except:
        return jsonify({"error": "Error processing uploaded image."}), 500
    
    return jsonify({"match": False, "matched_with": None})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
