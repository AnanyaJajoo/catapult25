import os
import cv2
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import threading # For locking during database updates

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'} # Add video extensions you want to support
DB_PREFIX = "video_db" # Prefix for your FAISS index and metadata files
MODEL_NAME = "google/siglip2-giant-opt-patch16-384" # Or your chosen model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # Adjust based on your GPU memory for embedding generation

# --- Initialize Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Model and Database (Load once at startup) ---
print(f"Loading model: {MODEL_NAME}...")
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")

# --- Database Lock ---
# To prevent race conditions when multiple requests try to add videos
db_lock = threading.Lock()

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_db_data(prefix):
    """Loads FAISS index and metadata."""
    index_path = f"{prefix}_index.faiss"
    metadata_path = f"{prefix}_metadata.pkl"
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print(f"Warning: Database files not found at prefix '{prefix}'. Returning empty.")
        # You might want to initialize an empty index/metadata here if needed
        return None, None
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        print(f"Database loaded: {index.ntotal} embeddings.")
        return index, metadata
    except Exception as e:
        print(f"Error loading database: {e}")
        return None, None

def save_db_data(index, metadata, prefix):
    """Saves FAISS index and metadata."""
    try:
        faiss.write_index(index, f"{prefix}_index.faiss")
        with open(f"{prefix}_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"Database saved with {index.ntotal} embeddings.")
    except Exception as e:
        print(f"Error saving database: {e}")

def extract_frames(video_path, n_frames):
    """Extracts every nth frame from a video."""
    # (Copied and adapted from create_db.py - ensure it matches your version)
    frames = []
    frame_numbers = []
    timestamps = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return [], [], []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or fps <= 0:
             print(f"Warning: Invalid metadata for video {video_path} (frames={total_frames}, fps={fps})")
             cap.release()
             return [], [], []

        target_frames = range(0, total_frames, n_frames)

        for frame_number in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_number / fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
            frame_numbers.append(frame_number)
            timestamps.append(timestamp)
        cap.release()
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return [], [], []
    return frames, frame_numbers, timestamps

def generate_embeddings(frames):
    """Generates embeddings for a list of frames."""
    # (Copied and adapted from create_db.py - ensure it matches your version)
    all_embeddings = []
    if not frames:
        return np.array([])
    try:
        with torch.no_grad():
            for i in range(0, len(frames), BATCH_SIZE):
                batch = frames[i:i+BATCH_SIZE]
                inputs = processor(images=batch, return_tensors="pt").to(DEVICE)
                outputs = model(**inputs)
                # --- Adjust based on your specific SigLIP model output ---
                # Option 1: If model output has 'image_embeds' directly
                # batch_embeddings = outputs.image_embeds
                # Option 2: If it's nested under vision_model_output
                batch_embeddings = outputs.vision_model_output.pooler_output
                # --- End Adjust ---
                all_embeddings.append(batch_embeddings.cpu().numpy())
        if not all_embeddings:
             return np.array([])
        return np.vstack(all_embeddings)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return np.array([])

def encode_text(text_query):
    """Encodes a text query using the SigLIP model."""
    try:
        # Note: SigLIP models process text and images differently.
        # Use the processor for text input.
        inputs = processor(text=[text_query], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # --- Adjust based on your specific SigLIP model output ---
            # Option 1: If model output has 'text_embeds' directly
            # text_embedding = outputs.text_embeds
            # Option 2: If it's nested under text_model_output
            text_embedding = outputs.text_model_output.pooler_output
            # --- End Adjust ---
        return text_embedding.cpu().numpy()
    except Exception as e:
        print(f"Error encoding text '{text_query}': {e}")
        return None

# --- Flask Routes ---

@app.route('/search', methods=['POST'])
def search_database():
    """
    Searches the database with a text query.
    Expects JSON: {"query": "search text", "k": 5}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')
    k = int(data.get('k', 5)) # Default to 5 results

    if not query:
        return jsonify({"error": "Missing 'query' in request"}), 400

    # Encode the text query
    text_embedding = encode_text(query)
    if text_embedding is None:
        return jsonify({"error": "Failed to encode text query"}), 500

    # Load database (read-only access, no lock needed)
    index, metadata = load_db_data(DB_PREFIX)
    if index is None or metadata is None:
         return jsonify({"error": "Database not loaded or not found"}), 500
    if index.ntotal == 0:
         return jsonify({"results": [], "message": "Database is empty"}), 200


    # Search the FAISS index
    try:
        distances, indices = index.search(text_embedding.astype('float32'), k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return jsonify({"error": "Error during database search"}), 500

    # Format results
    results = []
    if len(indices) > 0:
        for i, idx in enumerate(indices[0]):
            if idx < 0: continue # FAISS can return -1 for no results
            meta = metadata.get(int(idx))
            if meta:
                results.append({
                    "distance": float(distances[0][i]),
                    "score": 1.0 / (1.0 + float(distances[0][i])), # Example similarity score
                    "metadata": meta
                })

    return jsonify({"results": results})

@app.route('/add_video', methods=['POST'])
def add_video_to_database():
    """
    Adds a new video to the database.
    Expects form data: 'video': file, 'n_frames': int (optional)
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file part in the request"}), 400

    file = request.files['video']
    n_frames = int(request.form.get('n_frames', 30)) # Default to every 30th frame

    if file.filename == '':
        return jsonify({"error": "No selected video file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(video_path)
            print(f"Video saved temporarily to {video_path}")

            # --- Process Video and Update DB (CRITICAL SECTION) ---
            with db_lock:
                print("Acquired DB lock for update...")
                # Load current DB state *inside the lock*
                index, metadata = load_db_data(DB_PREFIX)
                if index is None or metadata is None:
                    # Handle case where DB doesn't exist yet - initialize
                    print("Initializing new database.")
                    # Determine embedding dimension (important!)
                    # You might need to run a dummy input through the model once
                    # Or hardcode it if you know it (e.g., 768 for base models)
                    dummy_img = Image.new('RGB', (224, 224))
                    dummy_emb = generate_embeddings([dummy_img])
                    if dummy_emb.size == 0:
                         # Clean up uploaded file before erroring
                         if os.path.exists(video_path): os.remove(video_path)
                         return jsonify({"error": "Could not determine embedding dimension"}), 500
                    dimension = dummy_emb.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    metadata = {}

                # 1. Extract Frames
                frames, frame_numbers, timestamps = extract_frames(video_path, n_frames)
                if not frames:
                    print(f"No frames extracted from {filename}")
                    # Clean up uploaded file
                    if os.path.exists(video_path): os.remove(video_path)
                    return jsonify({"error": f"No frames extracted from video {filename}"}), 400

                # 2. Generate Embeddings
                new_embeddings = generate_embeddings(frames)
                if new_embeddings.size == 0:
                    print(f"Failed to generate embeddings for {filename}")
                     # Clean up uploaded file
                    if os.path.exists(video_path): os.remove(video_path)
                    return jsonify({"error": f"Failed to generate embeddings for {filename}"}), 500

                # 3. Update Metadata and Index
                start_idx = index.ntotal # Index of the first new embedding
                index.add(new_embeddings.astype('float32')) # Add to FAISS index

                relative_video_path = filename # Store relative path or identifier
                for i, (frame_num, timestamp) in enumerate(zip(frame_numbers, timestamps)):
                    metadata[start_idx + i] = {
                        "video_path": relative_video_path, # Use filename or a relative path
                        "frame_number": frame_num,
                        "timestamp": timestamp
                    }

                # 4. Save Updated DB *inside the lock*
                save_db_data(index, metadata, DB_PREFIX)
                print("Released DB lock.")
            # --- End Critical Section ---

            # Optional: Clean up the uploaded file after processing
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Removed temporary file {video_path}")

            return jsonify({
                "message": f"Video '{filename}' added successfully",
                "frames_added": len(frames),
                "total_embeddings": index.ntotal
            }), 201 # 201 Created status

        except Exception as e:
            print(f"Error processing video {filename}: {e}")
             # Clean up uploaded file on error
            if os.path.exists(video_path): os.remove(video_path)
            return jsonify({"error": f"Failed to process video: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

# --- Run the App ---
if __name__ == '__main__':
    # Consider security implications of running on 0.0.0.0
    # For development: host='127.0.0.1'
    # For access within a local network: host='0.0.0.0'
    app.run(debug=True, host='0.0.0.0', port=5000) # Use debug=False in production