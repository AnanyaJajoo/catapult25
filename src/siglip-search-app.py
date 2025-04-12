import os
import cv2
import gradio as gr
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
from create_db import VideoEmbeddingDatabase

# Initialize the model and database
def load_model_and_db(db_prefix="video_db", model_name="google/siglip2-giant-opt-patch16-384"):
    db = VideoEmbeddingDatabase.load(db_prefix, model_name=model_name)
    return db

# Function to encode text queries
def encode_text(db, text_query):
    # use get_text_features instead of get_image_features
    inputs = db.tokenizer([text_query], return_tensors="pt", padding="max_length", truncation=True).to(db.device)
    with torch.no_grad():
        text_embedding = db.model.get_text_features(**inputs)
    return text_embedding.detach().cpu().numpy()

# Search function
def search_with_text(text_query, db, k=5):
    if not text_query.strip():
        return None, []
    
    # Encode the text query
    text_embedding = encode_text(db, text_query)
    
    # Search the FAISS index
    distances, indices = db.index.search(text_embedding.astype('float32'), k)

    print('[DEBUG]: distances:', distances)
    print('[DEBUG]: indices:', indices)
    
    results = []
    result_images = []
    
    for i, idx in enumerate(indices[0]):
        metadata = db.metadata[int(idx)]
        video_path = metadata["video_path"]
        frame_number = metadata["frame_number"]
        timestamp = metadata["timestamp"]
        
        # Extract the frame from the video
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        print()
        
        if ret:
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_images.append(frame_rgb)
            
            # Format result text
            video_name = os.path.basename(video_path)
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            
            result_text = f"Video: {video_name}<br>"
            result_text += f"Time: {minutes:02d}:{seconds:02d}<br>"
            result_text += f"Frame: {frame_number}<br>"
            result_text += f"Score: {1.0/(1.0+distances[0][i]):.2f}"

            print('[DEBUG] result_text:', result_text)
            
            results.append(result_text)
    
    results = []
    for (img, html) in zip(result_images, results):
        if img is not None:
            img = Image.fromarray(img)
        results.append((img, html))

    return result_images, results

# Main Gradio interface
def create_interface(db_prefix="video_db", model_name="google/siglip2-giant-opt-patch16-384"):
    print("Loading model and database...")
    db = load_model_and_db(db_prefix, model_name)
    print("Ready!")
    
    # Update your search function to return the correct number of outputs
    def search(query, top_k):
        images, captions = search_with_text(query, db, k=top_k)
        if images is None:
            return None, "", "", "", "", ""
        
        # Pad with None and empty strings if fewer than 5 results
        images = images[:5]  # Limit to first 5
        while len(images) < 5:
            images.append(None)
        
        captions = captions[:5]  # Limit to first 5
        while len(captions) < 5:
            captions.append("")
        
        # Return gallery images and individual HTML contents
        return images, captions[0], captions[1], captions[2], captions[3], captions[4]
    
    with gr.Blocks(title="Video Frame Semantic Search") as demo:
        gr.Markdown("# Video Frame Search")
        gr.Markdown("Enter a text description to find matching frames in the video database.")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Text Query", placeholder="Enter your search query...")
                top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of results")
                search_button = gr.Button("Search")
            
        with gr.Row():
            gallery = gr.Gallery(
                label="Search Results",
                show_label=True,
                elem_id="gallery",
                columns=5,
                height=400
            )
        
        with gr.Row():
            result_texts = [gr.HTML(label=f"Result {i+1}") for i in range(5)]
        
        search_button.click(
            fn=search,
            inputs=[query_input, top_k],
            outputs=[gallery] + result_texts  # where result_texts is a list of 5 HTML components
        )
    
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-prefix", type=str, default="video_db", help="Database prefix")
    parser.add_argument("--model-name", type=str, default="google/siglip2-giant-opt-patch16-384", 
                        help="Model name")
    parser.add_argument("--share", action="store_true", help="Share the Gradio interface")
    
    args = parser.parse_args()
    
    demo = create_interface(args.db_prefix, args.model_name)
    demo.launch(share=args.share)