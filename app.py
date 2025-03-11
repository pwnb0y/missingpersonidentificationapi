from flask import Flask, jsonify, request
from flask_cors import CORS
from deepface import DeepFace
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp"
ENCODINGS_FILE = "encodings.pkl"
CREDENTIALS_FILE = "credentials.json"
DRIVE_FOLDER_ID_KNOWN = "17zt840iXARAn9sQmDWmqpy4pdSTQLRdS"
ENCODINGS_FOLDER_ID = "1g1qZ3X6Yq6l_PQh79pVGbYCe7xwvkiyR"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

SCOPES = ["https://www.googleapis.com/auth/drive"]
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
drive_service = build("drive", "v3", credentials=credentials)

known_encodings = {}

def list_drive_files(folder_id):
    """List all files in a Google Drive folder."""
    query = f"'{folder_id}' in parents"
    results = drive_service.files().list(q=query, fields="files(id, name)").execute()
    return {file["id"]: file["name"] for file in results.get("files", [])}

def get_local_files():
    """Get a list of locally stored images."""
    return set(os.listdir(TEMP_FOLDER))

def download_new_images():
    """Download only new images that are missing locally."""
    current_files = list_drive_files(DRIVE_FOLDER_ID_KNOWN)
    local_files = get_local_files()

    new_files = {fid: fname for fid, fname in current_files.items() if fname not in local_files}

    if new_files:
        print(f"🆕 {len(new_files)} new images detected! Downloading...")
        for file_id, filename in new_files.items():
            file_path = os.path.join(TEMP_FOLDER, filename)
            request = drive_service.files().get_media(fileId=file_id)
            with open(file_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            print(f"✅ Downloaded: {filename}")
        encode_new_images()
    else:
        print("✅ No new images found. Everything is up to date.")

def load_encodings():
    """Load encodings from a file if available."""
    global known_encodings
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            known_encodings = pickle.load(f)

def save_encodings():
    """Save encodings to a file and upload to Google Drive."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_encodings, f)
    upload_file_to_drive(ENCODINGS_FILE, ENCODINGS_FOLDER_ID, ENCODINGS_FILE)

def upload_file_to_drive(file_path, folder_id, filename):
    """Upload or update a file on Google Drive."""
    file_metadata = {"name": filename, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="application/octet-stream")

    query = f"'{folder_id}' in parents and name='{filename}'"
    existing_files = drive_service.files().list(q=query, fields="files(id)").execute().get("files", [])

    if existing_files:
        file_id = existing_files[0]["id"]
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        print(f"✅ Updated: {filename} on Google Drive.")
    else:
        drive_service.files().create(body=file_metadata, media_body=media).execute()
        print(f"✅ Uploaded: {filename} to Google Drive.")

def encode_new_images():
    """Encode only new images and update encodings."""
    global known_encodings
    new_encodings = {}

    for filename in os.listdir(TEMP_FOLDER):
        file_path = os.path.join(TEMP_FOLDER, filename)

        if filename in known_encodings:  # Skip already encoded images
            continue

        try:
            embeddings = DeepFace.represent(img_path=file_path, model_name="Facenet", enforce_detection=False)
            if embeddings:
                new_encodings[filename] = np.array(embeddings[0]["embedding"])
                print(f"✅ Encoded: {filename}")
        except Exception as e:
            print(f"⚠️ Error encoding {filename}: {e}")

    if new_encodings:
        known_encodings.update(new_encodings)
        save_encodings()
        print("✅ Encoding update completed.")

@app.route("/match", methods=["POST"])
def match_uploaded_face():
    """Match uploaded face against stored encodings."""
    global known_encodings

    download_new_images()  # Ensure latest images are downloaded and encoded

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        result = DeepFace.find(img_path=file_path, db_path=TEMP_FOLDER, model_name="Facenet", enforce_detection=False)
        
        if isinstance(result, list) and len(result) > 0 and not result[0].empty:
            matched_name = os.path.basename(result[0]['identity'].values[0])
            return jsonify({"match": True, "matched_with": matched_name})
        else:
            return jsonify({"match": False, "message": "No match found."})
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": "Error processing uploaded image.", "details": str(e)}), 500
    finally:
        os.remove(file_path)

@app.route("/update", methods=["GET"])
def manual_update():
    """Manually trigger an update: download new images and encode them."""
    download_new_images()
    return jsonify({"message": "New images downloaded and encoded."})

if __name__ == "__main__":
    load_encodings()  # Load previous encodings if available
    download_new_images()  # Download only missing images and encode them
    app.run(debug=True, host="0.0.0.0", port=5000)
