import os
import glob
import argparse
import cv2
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from tqdm import tqdm
import pickle

def gen_type_tree(structure):
    """
    Generate a type tree from the given structure.
    """
    if isinstance(structure, dict):
        return {key: gen_type_tree(value) for key, value in structure.items()}
    elif isinstance(structure, list):
        return [gen_type_tree(item) for item in structure]
    else:
        return type(structure).__name__
    
def print_type_tree(structure, indent=0):
    """
    Print the type tree of the given structure.
    """
    if isinstance(structure, dict):
        for key, value in structure.items():
            print(" " * indent + f"{key}: {type(value).__name__}")
            print_type_tree(value, indent + 2)
    elif isinstance(structure, list):
        for index, item in enumerate(structure):
            print(" " * indent + f"[{index}]: {type(item).__name__}")
            print_type_tree(item, indent + 2)
    else:
        print(" " * indent + f"{type(structure).__name__}")

def extract_frames(video_path, n_frames=30):
    """Extract every nth frame from a video using direct seeking."""
    frames = []
    frame_numbers = []
    timestamps = []
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    # Calculate which frames to extract
    target_frames = range(0, total_frames, n_frames)
        
    with tqdm(total=len(target_frames), desc=f"Extracting frames from {os.path.basename(video_path)}") as pbar:
        for frame_number in target_frames:
            # Set position to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break
                
                # Process the frame
            timestamp = frame_number / fps
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
                
            frames.append(pil_image)
            frame_numbers.append(frame_number)
            timestamps.append(timestamp)
            pbar.update(1)
        
    cap.release()
    return frames, frame_numbers, timestamps

def extract_all_videos(video_paths):
            import multiprocessing
            with multiprocessing.Pool() as pool:
                results = pool.map(extract_frames, video_paths)
            return results

class VideoEmbeddingDatabase:
    def __init__(self, model_name="google/siglip2-giant-opt-patch16-384", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.metadata = {}  # {index: (video_path, frame_number, timestamp)}
        self.index = None

        print(f"Using device: {self.device}, with properties:")
        print(torch.cuda.get_device_properties(0) if self.device == "cuda" else "CPU")



    def generate_embeddings(self, frames, batch_size=1):
        """Generate embeddings for a list of frames using SigLIP2."""
        # we know dim is [n, 1536]
        embeddings = np.zeros((len(frames), 1536))
        
        # for i in range(0, len(frames), batch_size):
        for i in tqdm(range(0, len(frames), batch_size), desc="Generating embeddings"):
            batch = frames[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                embedding = self.model.get_image_features(**inputs)
                embedding = embedding.detach().cpu().numpy()
                if batch_size == 1:
                    embeddings[i] = embedding
                else:
                    embeddings[i:i+batch_size] = embedding
        
        return embeddings

    def build_database(self, video_dir, n_frames=30):
        """Build a FAISS index from videos in the specified directory."""
        ''' and in all subdirectories '''
        video_paths = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
        if not video_paths:
            raise ValueError(f"No MP4 videos found in {video_dir}")
                
        all_embeddings = []
        index_counter = 0
        
        # Extract frames from all videos
        results = extract_all_videos(video_paths)
        for video_path, (frames, frame_numbers, timestamps) in zip(video_paths, results):

            if not frames:
                print(f"No frames extracted from {video_path}")
                continue
                
            all_embeddings.extend(self.generate_embeddings(frames))
            
            # Store metadata
            for i, (frame_num, timestamp) in enumerate(zip(frame_numbers, timestamps)):
                self.metadata[index_counter + i] = {
                    "video_path": video_path,
                    "frame_number": frame_num,
                    "timestamp": timestamp
                }
            
            index_counter += len(frames)
        
        if not all_embeddings:
            raise ValueError("No embeddings were generated")
            
        # Concatenate all embeddings
        all_embeddings_temp = np.vstack(all_embeddings)

        # copy
        all_embeddings = np.zeros_like(all_embeddings_temp)
        all_embeddings = all_embeddings_temp.copy()

        # Create FAISS index
        dimension = all_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(all_embeddings)
        
        print(f"Database built with {self.index.ntotal} embeddings")
        return self.index, self.metadata

    def save(self, path_prefix):
        """Save the database and metadata."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path_prefix}_index.faiss")
        
        # Save metadata
        with open(f"{path_prefix}_metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"Database saved to {path_prefix}_index.faiss and {path_prefix}_metadata.pkl")

    @classmethod
    def load(cls, path_prefix, model_name="google/siglip2-giant-opt-patch16-384"):
        """Load a saved database."""
        instance = cls(model_name=model_name)
        
        # Load FAISS index
        instance.index = faiss.read_index(f"{path_prefix}_index.faiss")
        
        # Load metadata
        with open(f"{path_prefix}_metadata.pkl", "rb") as f:
            instance.metadata = pickle.load(f)
        
        print(f"Database loaded with {instance.index.ntotal} embeddings")
        return instance

    def query(self, image, k=5):
        """Query the database with an image."""
        if self.index is None:
            raise ValueError("No index to query")
        
        # Process the image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Generate embedding
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            embedding = outputs.vision_model_output.pooler_output.cpu().numpy()
        
        # Query the index
        distances, indices = self.index.search(embedding.astype('float32'), k)
        
        # Get metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "distance": distances[0][i],
                "metadata": self.metadata[int(idx)]
            })
        
        return results
    



def main():
    parser = argparse.ArgumentParser(description="Generate video embedding database using SigLIP2")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory containing MP4 videos")
    parser.add_argument("--output", type=str, default="video_db", help="Output prefix for database files")
    parser.add_argument("--n-frames", type=int, default=30, help="Process every nth frame")
    parser.add_argument("--model-name", type=str, default="google/siglip2-giant-opt-patch16-384", help="Model name")
    
    args = parser.parse_args()
    
    # Create and build the database
    db = VideoEmbeddingDatabase(model_name=args.model_name)
    db.build_database(args.video_dir, n_frames=args.n_frames)
    
    # Save the database
    db.save(args.output)
    
    print("Database generation complete!")


if __name__ == "__main__":
    main()