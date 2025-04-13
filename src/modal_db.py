import modal
import os
import glob
import argparse
import cv2
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import pickle
import time # Added for potential unique naming

# Define Modal stub
stub = modal.Stub("video-embedding-api")

# Define Modal image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "pillow",
    "opencv-python",
    "tqdm",
    "faiss-cpu",
    "numpy"
)

# Mount cloud storage if needed
VOLUME = modal.Volume.from_name("video-data-volume")
VIDEO_DIR = "/videos/videos" # Primarily for build_database
OUTPUT_DIR = "/output" # For database files and temporary uploads
DB_LOCK = modal.Lock() # Lock to prevent concurrent writes to the database files

# Create a class to wrap your database functionality
@stub.cls(
    image=image,
    gpu="H100",  # Choose appropriate GPU type
    volumes={VIDEO_DIR: VOLUME, OUTPUT_DIR: VOLUME} # Ensure OUTPUT_DIR is mounted
)
class VideoEmbeddingAPI:
    def __enter__(self):
        # This runs when the container starts
        # Determine embedding dimension dynamically or hardcode if known
        # Example: Hardcoding based on model name
        self.embedding_dimension = 1536 # Dimension for siglip2-giant-opt-patch16-384
        self.model = AutoModel.from_pretrained("google/siglip2-giant-opt-patch16-384").to("cuda")
        self.processor = AutoProcessor.from_pretrained("google/siglip2-giant-opt-patch16-384")
        self.model.eval()

    # Implement your methods as Modal functions
    def extract_frames(self, video_path, n_frames):
        """Extract every nth frame from a video."""
        # ... existing extract_frames code ...
        # Ensure this function returns frames, frame_numbers, timestamps
        frames = []
        frame_numbers = []
        timestamps = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return frames, frame_numbers, timestamps

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = n_frames

            for frame_num in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    # Convert frame from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    frame_numbers.append(frame_num)
                    timestamps.append(frame_num / fps if fps > 0 else 0)
                else:
                    # Break if reading fails, might be end of video or error
                    break
            cap.release()
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            # Ensure lists are returned even on error
            return [], [], []

        return frames, frame_numbers, timestamps
        # ... end of extract_frames code ...

    def generate_embeddings(self, frames, batch_size=8):
        """Generate embeddings for a list of frames using SigLIP2."""
        # ... existing generate_embeddings code ...
        all_embeddings = []
        if not frames:
            return np.array([])

        try:
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i+batch_size]
                inputs = self.processor(images=batch_frames, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use image_embeds directly if available and correct, otherwise pooler_output
                    # For SigLIP, image_embeds might be preferred if pre-pooling
                    # Check model documentation or output structure
                    # Assuming pooler_output is the final embedding per image
                    batch_embeddings = outputs.image_embeds # Or outputs.pooler_output depending on model version/config
                    if batch_embeddings is None and hasattr(outputs, 'pooler_output'):
                         batch_embeddings = outputs.pooler_output
                    elif batch_embeddings is None and hasattr(outputs, 'vision_model_output') and hasattr(outputs.vision_model_output, 'pooler_output'):
                         batch_embeddings = outputs.vision_model_output.pooler_output


                    if batch_embeddings is not None:
                         all_embeddings.extend(batch_embeddings.cpu().numpy())
                    else:
                         print(f"Warning: Could not extract embeddings for batch starting at index {i}")


        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.array([]) # Return empty array on error

        return np.array(all_embeddings)
        # ... end of generate_embeddings code ...

    @modal.method()
    def build_database(self, video_dir=VIDEO_DIR, n_frames=30, output_prefix=f"{OUTPUT_DIR}/video_db"):
        """Build a FAISS index from videos in the specified directory."""
        # ... existing build_database code ...
        # Ensure it uses the class methods self.extract_frames and self.generate_embeddings
        # Ensure it saves index and metadata correctly
        video_paths = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True) \
                    + glob.glob(os.path.join(video_dir, "**", "*.avi"), recursive=True) \
                    + glob.glob(os.path.join(video_dir, "**", "*.mov"), recursive=True) # Add more extensions if needed

        if not video_paths:
            print(f"Warning: No video files found in {video_dir}")
            # Optionally create empty files if needed downstream
            # dimension = self.embedding_dimension
            # index = faiss.IndexFlatL2(dimension)
            # metadata = {}
            # faiss.write_index(index, f"{output_prefix}_index.faiss")
            # with open(f"{output_prefix}_metadata.pkl", "wb") as f:
            #     pickle.dump(metadata, f)
            return {"status": "skipped", "message": "No videos found"}

        all_embeddings_list = []
        metadata = {}
        index_counter = 0

        # Use tqdm for progress indication
        for video_path in tqdm(video_paths, desc="Processing videos"):
            print(f"Processing {video_path}")
            # Use self methods
            frames, frame_numbers, timestamps = self.extract_frames(video_path, n_frames)

            if not frames:
                print(f"No frames extracted from {video_path}, skipping.")
                continue

            # Use self methods
            embeddings = self.generate_embeddings(frames)
            if embeddings.size == 0:
                 print(f"No embeddings generated for {video_path}, skipping.")
                 continue

            all_embeddings_list.append(embeddings)

            # Store metadata
            relative_path = os.path.relpath(video_path, video_dir)
            for i, (frame_num, timestamp) in enumerate(zip(frame_numbers, timestamps)):
                metadata[index_counter + i] = {
                    "video_path": relative_path,
                    "frame_number": frame_num,
                    "timestamp": timestamp
                }

            index_counter += len(frames)

        if not all_embeddings_list:
             print("No embeddings generated from any video.")
             # Create empty files as above if needed
             return {"status": "error", "message": "No embeddings generated"}

        # Concatenate all embeddings
        all_embeddings = np.concatenate(all_embeddings_list, axis=0)

        # Create FAISS index
        dimension = all_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(all_embeddings.astype('float32'))

        # Save the index and metadata
        index_file = f"{output_prefix}_index.faiss"
        metadata_file = f"{output_prefix}_metadata.pkl"
        os.makedirs(os.path.dirname(index_file), exist_ok=True) # Ensure dir exists

        with DB_LOCK: # Use lock for writing
            faiss.write_index(index, index_file)
            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)

        print(f"Database build complete. Index: {index_file}, Metadata: {metadata_file}")
        return {"status": "success", "embeddings_count": index.ntotal}
        # ... end of build_database code ...

    @modal.method()
    def add_video(self, video_path_on_volume, n_frames=30, output_prefix=f"{OUTPUT_DIR}/video_db"):
        """Adds a single video to the existing FAISS index and metadata."""
        index_file = f"{output_prefix}_index.faiss"
        metadata_file = f"{output_prefix}_metadata.pkl"

        # Acquire lock to prevent concurrent read/modify/write operations
        with DB_LOCK:
            print(f"Acquired lock for adding video: {video_path_on_volume}")
            # --- Load existing index and metadata ---
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                try:
                    index = faiss.read_index(index_file)
                    with open(metadata_file, "rb") as f:
                        metadata = pickle.load(f)
                    start_index = index.ntotal
                    # Verify dimension matches
                    if index.d != self.embedding_dimension:
                         raise ValueError(f"Index dimension ({index.d}) does not match model dimension ({self.embedding_dimension})")
                    print(f"Loaded existing DB. Index size: {start_index}, Metadata size: {len(metadata)}")
                except Exception as e:
                    print(f"Error loading existing database files from {output_prefix}: {e}")
                    raise # Re-raise the exception

            else:
                # If DB doesn't exist, cannot add to it. build_database should run first.
                message = f"Database files not found at {output_prefix}. Cannot add video. Run build_database first."
                print(message)
                raise FileNotFoundError(message)

            # --- Process the new video ---
            print(f"Processing new video: {video_path_on_volume}")
            frames, frame_numbers, timestamps = self.extract_frames(video_path_on_volume, n_frames)

            if not frames:
                print(f"No frames extracted from {video_path_on_volume}, skipping add.")
                # Release lock implicitly by exiting 'with' block
                return {"status": "skipped", "message": "No frames extracted."}

            embeddings = self.generate_embeddings(frames)
            if embeddings.size == 0:
                 print(f"No embeddings generated for {video_path_on_volume}, skipping add.")
                 return {"status": "skipped", "message": "No embeddings generated."}

            new_embeddings_np = embeddings.astype('float32')

            # --- Update metadata ---
            # Use basename or a relative path meaningful to the web app context
            relative_video_path = os.path.basename(video_path_on_volume)
            num_new_embeddings = len(frames)
            print(f"Generated {num_new_embeddings} new embeddings.")
            for i in range(num_new_embeddings):
                current_metadata_index = start_index + i
                metadata[current_metadata_index] = {
                    "video_path": relative_video_path,
                    "frame_number": frame_numbers[i],
                    "timestamp": timestamps[i]
                }

            # --- Add embeddings to index ---
            index.add(new_embeddings_np)
            print(f"Added {num_new_embeddings} embeddings to FAISS index. New total: {index.ntotal}")

            # --- Save updated index and metadata (overwrite) ---
            try:
                faiss.write_index(index, index_file)
                with open(metadata_file, "wb") as f:
                    pickle.dump(metadata, f)
                print(f"Successfully saved updated index and metadata to {output_prefix}")
            except Exception as e:
                print(f"Error saving updated database files: {e}")
                # Consider how to handle partial updates - maybe revert? For now, just raise.
                raise

        # Lock released automatically here
        print(f"Released lock for {video_path_on_volume}")
        return {"status": "success", "added_count": num_new_embeddings, "total_embeddings": index.ntotal}


    @modal.method()
    def query(self, text_query, db_prefix=f"{OUTPUT_DIR}/video_db", k=5):
        """Query the database with text."""
        index_file = f"{db_prefix}_index.faiss"
        metadata_file = f"{db_prefix}_metadata.pkl"

        # No lock needed for reading unless build/add can happen concurrently
        # If concurrent writes are possible, consider a read lock or retries
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
             return {"error": "Database not found. Please build or add videos first."}

        try:
            # Load the index and metadata
            index = faiss.read_index(index_file)
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            if index.ntotal == 0:
                 return {"results": [], "message": "Database is empty."}

            # Process the text query
            inputs = self.processor(text=[text_query], return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Adjust based on actual model output structure for text embeddings
                embedding = outputs.text_embeds # Or outputs.pooler_output etc.
                if embedding is None and hasattr(outputs, 'text_model_output') and hasattr(outputs.text_model_output, 'pooler_output'):
                     embedding = outputs.text_model_output.pooler_output

                if embedding is None:
                     raise ValueError("Could not extract text embedding from model output.")

                embedding_np = embedding.cpu().numpy()

            # Query the index
            distances, indices = index.search(embedding_np.astype('float32'), k)

            # Get metadata for results
            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if idx != -1: # FAISS uses -1 for invalid indices
                        metadata_key = int(idx)
                        if metadata_key in metadata:
                            results.append({
                                "distance": float(distances[0][i]),
                                "metadata": metadata[metadata_key]
                            })
                        else:
                            print(f"Warning: Index {metadata_key} not found in metadata.")

            return {"results": results}
        except Exception as e:
             print(f"Error during query: {e}")
             return {"error": f"Query failed: {e}"}


# --- Web Endpoints ---

# Endpoint to add a single video
@stub.function(
    image=image,
    volumes={OUTPUT_DIR: VOLUME}, # Need volume access to save temp file & call add_video
    timeout=600 # Increase timeout for potentially long uploads/processing
    # secrets=[modal.Secret.from_name("my-auth-secret")] # Add if auth needed
)
@modal.web_endpoint(method="POST")
async def add_video_endpoint(request: modal.Request):
    """
    API endpoint to add a video.
    Expects POST request with multipart/form-data containing:
    - 'video_file': The video file to add.
    - 'n_frames' (optional): Frame extraction interval (default: 30).
    - 'db_prefix' (optional): Path prefix for db files on volume (default: /output/video_db).
    """
    start_time = time.time()
    try:
        form = await request.form()
        if "video_file" not in form:
            return modal.Response(
                {"error": "Missing 'video_file' in form data"},
                status_code=400
            )

        video_data = await form["video_file"].read()
        original_filename = form["video_file"].filename or f"upload_{int(time.time())}.mp4"
        n_frames = int(form.get("n_frames", 30))
        db_prefix = form.get("db_prefix", f"{OUTPUT_DIR}/video_db")

        # Define a temporary path on the shared volume within OUTPUT_DIR
        # Using a subdirectory and unique name helps avoid collisions
        upload_dir = os.path.join(OUTPUT_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        temp_video_path = os.path.join(upload_dir, f"{modal.current_function_call.id}_{original_filename}")

        # Save the uploaded file to the volume
        with open(temp_video_path, "wb") as f:
            f.write(video_data)
        print(f"Temporarily saved uploaded video to: {temp_video_path}")
        save_time = time.time()
        print(f"Time to save upload: {save_time - start_time:.2f}s")


        # Call the class method to process the video from the volume
        api = VideoEmbeddingAPI()
        result = api.add_video.remote(
            video_path_on_volume=temp_video_path,
            n_frames=n_frames,
            output_prefix=db_prefix
        )

        process_time = time.time()
        print(f"Time for add_video call: {process_time - save_time:.2f}s")
        print(f"add_video result: {result}")

        # Return the result from add_video
        status_code = 200 if result.get("status") == "success" else 500
        if result.get("status") == "skipped":
             status_code = 200 # Not an error, just nothing added

        return modal.Response(result, status_code=status_code)

    except FileNotFoundError as e:
         # Specific handling for DB not found error from add_video
         print(f"Error: {e}")
         return modal.Response({"error": str(e)}, status_code=404) # 404 Not Found seems appropriate
    except Exception as e:
        print(f"Error processing add_video request for {original_filename}: {e}")
        import traceback
        traceback.print_exc()
        return modal.Response(
            {"error": f"Failed to process video: {str(e)}"},
            status_code=500
        )
    finally:
        # Clean up the temporary file if it exists
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                print(f"Removed temporary file: {temp_video_path}")
            except OSError as e:
                # Log error but don't fail the request because of cleanup issue
                print(f"Warning: Error removing temporary file {temp_video_path}: {e}")
        end_time = time.time()
        print(f"Total add_video_endpoint duration: {end_time - start_time:.2f}s")


# Existing search endpoint
@stub.function(image=image, volumes={OUTPUT_DIR: VOLUME}) # Ensure volume access for reading DB
@modal.web_endpoint(method="POST")
async def search_videos(request: modal.Request): # Use modal.Request for consistency
    """API endpoint to search videos using text."""
    try:
        data = await request.json()
        text_query = data.get("query")
        if not text_query:
             return modal.Response({"error": "Missing 'query' in request body"}, status_code=400)

        k = int(data.get("k", 5))
        db_prefix = data.get("db_prefix", f"{OUTPUT_DIR}/video_db") # Allow specifying db prefix

        api = VideoEmbeddingAPI()
        # Use .remote() for calling class methods
        results = api.query.remote(text_query, db_prefix=db_prefix, k=k)

        # Check if query returned an error structure
        if isinstance(results, dict) and "error" in results:
             status_code = 404 if "not found" in results["error"].lower() else 500
             return modal.Response(results, status_code=status_code)

        return modal.Response({"results": results}) # Assuming query returns the list directly now

    except Exception as e:
        print(f"Error in search_videos endpoint: {e}")
        import traceback
        traceback.print_exc()
        return modal.Response({"error": f"Search failed: {str(e)}"}, status_code=500)


@stub.local_entrypoint()
def main(action: str = "build", video_dir: str = VIDEO_DIR, output_prefix: str = f"{OUTPUT_DIR}/video_db", n_frames: int = 30, video_path: str = None):
    """Command-line entrypoint for managing the database."""
    api = VideoEmbeddingAPI()

    if action == "build":
        print(f"Building database from videos in: {video_dir}")
        print(f"Output prefix: {output_prefix}")
        print(f"Frame interval: {n_frames}")
        result = api.build_database.remote(video_dir=video_dir, n_frames=n_frames, output_prefix=output_prefix)
        print(f"Database build result: {result}")
    elif action == "add":
        if not video_path or not os.path.exists(video_path):
             print(f"Error: Video path '{video_path}' not provided or does not exist locally for action 'add'.")
             print("Note: 'add' via local_entrypoint assumes the file exists locally and uploads it.")
             # This local entrypoint version would need to upload the file to the volume first,
             # similar to how the web endpoint does, or assume video_path is already on the volume.
             # For simplicity, let's assume it needs to be uploaded.
             if video_path and os.path.exists(video_path):
                  target_path_on_volume = os.path.join(OUTPUT_DIR, "cli_uploads", os.path.basename(video_path))
                  print(f"Uploading {video_path} to {target_path_on_volume}...")
                  os.makedirs(os.path.dirname(target_path_on_volume), exist_ok=True)
                  # This requires modal client interaction with volumes, which is complex.
                  # Easier to instruct user to use the web endpoint or manually place file on volume.
                  print("Error: Adding via local_entrypoint requires manual file placement on the volume first.")
                  print(f"Please place '{video_path}' onto the volume (e.g., at '{target_path_on_volume}') and then run add.")
                  # Or implement modal volume put here if needed.
                  # Example (conceptual):
                  # with open(video_path, "rb") as f:
                  #     VOLUME.put_file(f, target_path_on_volume) # This syntax might vary
                  # print("Upload complete.")
                  # result = api.add_video.remote(video_path_on_volume=target_path_on_volume, n_frames=n_frames, output_prefix=output_prefix)
                  # print(f"Add video result: {result}")
             return # Exit if path invalid

        print(f"Adding video: {video_path}")
        print(f"Output prefix: {output_prefix}")
        print(f"Frame interval: {n_frames}")
        # Assuming video_path is ALREADY on the volume for this CLI example path
        result = api.add_video.remote(video_path_on_volume=video_path, n_frames=n_frames, output_prefix=output_prefix)
        print(f"Add video result: {result}")

    else:
        print(f"Unknown action: {action}. Use 'build' or 'add'.")