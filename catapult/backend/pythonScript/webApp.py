import cv2
import os
import asyncio
from inference_sdk import InferenceHTTPClient
import whisper
import pickle
from pydub import AudioSegment, silence
import subprocess
import mediapipe as mp
import numpy as np
from moviepy import AudioFileClip, VideoFileClip
from flask import Flask
from groq import Groq
import base64
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize Pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


chatter = Groq()
def get_response(image_path):

    base64_input = encode_image(image_path)

    completion = chatter.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer format:\n\n<Yes> or <No>, <Yes> or <No>, <Yes> or <No>.\n\n[Brief Description 10 words].\n\nQuestions:\nhas anyone fallen in this image, are there any dangerous objects in this image, are there any scared emotions in this image - answer in yes or no.\n\n10 words max - Tell me whether there is anything going wrong in this image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_input}",
                        }
                    }
                ]
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content


# Function to calculate the Euclidean distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to calculate "size" based on shoulder width
def get_shoulder_width(keypoints):
    # 11 = left shoulder, 12 = right shoulder
    return calculate_distance(keypoints[11], keypoints[12])

# Function to detect rapid movement based on shoulder width
def detect_rapid_movement(prev_keypoints, curr_keypoints, visibility_threshold=0.5):
    total_movement = 0
    valid_keypoints = 0
    
    for i in range(len(prev_keypoints)):
        # Check the visibility for the current keypoint
        if curr_keypoints[i][2] < visibility_threshold:  # visibility threshold (z-value, which is 0-1)
            continue
        
        # Calculate the movement only for valid keypoints
        total_movement += calculate_distance(prev_keypoints[i][:2], curr_keypoints[i][:2])
        valid_keypoints += 1

    # If not enough valid keypoints were found, return False
    if valid_keypoints == 0:
        return False

    # Use shoulder width as the relative scale
    shoulder_width = get_shoulder_width(curr_keypoints)
    threshold = shoulder_width * 15.0

    # if total_movement > threshold:
        # print("🚨 Rapid movement detected!")

# Async function to extract audio from MP4 and save as MP3
async def extract_audio_from_mp4(mp4_file_path, mp3_file_path):
    loop = asyncio.get_event_loop()

    # Run the ffmpeg command in a separate thread to avoid blocking the event loop
    await loop.run_in_executor(None, convert_mp4_to_mp3, mp4_file_path, mp3_file_path)

    print(f"Audio extracted and saved as {mp3_file_path}")

def convert_mp4_to_mp3(input_file, output_file=None):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".mp3"

    command = [
        "ffmpeg",
        "-y",            # Always yes: overwrite without asking
        "-i", input_file,
        "-q:a", "0",     # Best quality
        "-map", "a",
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversion complete: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")

lock = asyncio.Lock()
shared_list = [] # reset to 0

async def update_list_async(value):
    async with lock:
        # Safely update the shared list
        shared_list.append(value)

# Async function to load the model once
async def load_model_once():
    loop = asyncio.get_event_loop()
    
    # Check if the model is already saved (on the disk)
    if os.path.exists('processing_data/whisper_model.pkl'):
        with open('processing_data/whisper_model.pkl', 'rb') as f:
            model = pickle.load(f)
            print("Model loaded from pickle")
    else:
        # Load the model (blocking I/O wrapped in run_in_executor)
        model = await loop.run_in_executor(None, whisper.load_model, "small")
        with open('processing_data/whisper_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            print("Model saved to pickle")
    
    return model

# Async function to handle the image resizing and inference
async def process_dangerous_image(resizedImagePath, timestamp):
    # Perform inference asynchronously
    result = CLIENT.infer(resizedImagePath, model_id="dangerous-objects-dq94u/5")

    if(len(result) != 0):
        for i in range(len(result['predictions'])):
            # Extract bounding box info from the result
            x = int(result['predictions'][i]['x'])
            y = int(result['predictions'][i]['y'])
            w = int(result['predictions'][i]['width'])
            h = int(result['predictions'][i]['height'])
            dang_type = result['predictions'][i]['class']

            item = [timestamp, "dangerous", (x, y), (w, h), dang_type]

            await update_list_async(item)
            print(f"x: {x}, y: {y}, w: {w}, h: {h}")
            print("dangerous")

# Async function to handle the image resizing and inference
async def process_emotion_image(resizedImagePath, timestamp):
    # Perform inference asynchronously
    result = CLIENT.infer(resizedImagePath, model_id="facial-emotion-recognition/2")

    if(len(result) != 0):
        for i in range(len(result['predictions'])):
            # Extract bounding box info from the result
            x = int(result['predictions'][i]['x'])
            y = int(result['predictions'][i]['y'])
            w = int(result['predictions'][i]['width'])
            h = int(result['predictions'][i]['height'])
            emo_type = result['predictions'][i]['class']

            if(emo_type == "angry"):
                item = [timestamp, "emotion", (x, y), (w, h), emo_type]
                await update_list_async(item)
                print(f"x: {x}, y: {y}, w: {w}, h: {h}")
                print("emotion")


# Async function to handle the image resizing and inference
async def process_fall_image(resizedImagePath, timestamp):
    # Perform inference asynchronously
    result = CLIENT.infer(resizedImagePath, model_id="fall-detection-real/1")

    if(len(result) != 0):
        for i in range(len(result['predictions'])):
            # Extract bounding box info from the result
            x = int(result['predictions'][i]['x'])
            y = int(result['predictions'][i]['y'])
            w = int(result['predictions'][i]['width'])
            h = int(result['predictions'][i]['height'])
            fall_type = result['predictions'][i]['class']

            if(fall_type != 'standing'):
                item = [timestamp, "fell", (x, y), (w, h), fall_type]
                await update_list_async(item)
                print(f"x: {x}, y: {y}, w: {w}, h: {h}")
                print("fell")



# Async function to transcribe audio and find the timestamp for the word "help"
async def transcribe_audio(audio_file_url):
    model = await load_model_once()
    result = model.transcribe(audio_file_url)

    # Check for 'help' in the transcription and get timestamps if available
    for word_info in result['segments']:
        for word in word_info['tokens']:
            word_text = word_info['text']
            if 'help' in word_text.lower():
                timestamp = word_info['start']  # Get the start timestamp for the word
                item = [timestamp, "help"]
                await update_list_async(item)  # Update the list asynchronously
                print(f"Found 'help' at timestamp: {timestamp}")

# async function to find loud noises within a audio file
async def detect_loud_chunks(audio_file_path):
    # await extract_audio_from_mp4()
    loop = asyncio.get_event_loop()
    
    # Use run_in_executor to offload the blocking I/O operations
    audio = await loop.run_in_executor(None, AudioSegment.from_mp3, audio_file_path)

    # Detect non-silent chunks asynchronously
    nonsilent_ranges = await loop.run_in_executor(None, silence.detect_nonsilent, audio, 100, -10)

    # Print the detected ranges (in milliseconds)
    for start, end in nonsilent_ranges:
        # Convert start (milliseconds) to seconds
        timestamp = start / 1000.0
        
        # Create an item to update the list with timestamp and "loud" label
        item = [timestamp, "loud"]
        
        # Update the list asynchronously
        await update_list_async(item)
        
        # Print the detected loud sound range
        print(f"Loud sound from {start / 1000:.2f}s to {end / 1000:.2f}s")



# def extract_legal_5s_clip_with_audio(video_path, timestamp_sec, output_path):
#     # Use OpenCV to get FPS and total frame count
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Failed to open video.")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration_sec = total_frames / fps
#     cap.release()

#     clip_length_sec = 5

#     # Mirror your timestamp safety logic
#     start_time_sec = max(0, timestamp_sec - clip_length_sec / 2)

#     # Pull the clip back if it overflows past the end
#     if start_time_sec + clip_length_sec > duration_sec:
#         clip_length_sec = duration_sec - start_time_sec

#     # Final fallback check
#     start_time_sec = max(0, start_time_sec)

#     # Format for ffmpeg
#     start_str = f"{int(start_time_sec // 3600):02}:{int((start_time_sec % 3600) // 60):02}:{start_time_sec % 60:06.3f}"

#     # Create the ffmpeg command with forced container and codecs
#     cmd = [
#         "ffmpeg",
#         "-y",  # Overwrite without prompt
#         "-ss", start_str,
#         "-i", video_path,
#         "-t", str(clip_length_sec),
#         "-c:v", "libx264",  # Use H.264 codec for video encoding (MP4 compatible)
#         "-c:a", "aac",      # Use AAC codec for audio encoding (MP4 compatible)
#         "-strict", "experimental",  # Ensure FFmpeg uses the correct encoders
#         "-f", "mp4",        # Force format to mp4
#         "-movflags", "+faststart",  # Optimizes for web use (faster start time)
#         output_path
#     ]

#     # Execute the command
#     subprocess.run(cmd)
#     print(f"✅ Clip with audio saved to: {output_path}")


def extract_legal_5s_clip_with_audio(video_path, timestamp_sec, output_path):
    # Use OpenCV to get FPS and total frame count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    cap.release()

    clip_length_sec = 5

    # Mirror your timestamp safety logic
    start_time_sec = max(0, timestamp_sec - clip_length_sec / 2)

    # Pull the clip back if it overflows past the end
    if start_time_sec + clip_length_sec > duration_sec:
        clip_length_sec = duration_sec - start_time_sec

    # Final fallback check
    start_time_sec = max(0, start_time_sec)

    # Use OpenCV to extract frames
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time_sec * fps))

    # Create a VideoWriter to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Read and write frames for the duration of the clip
    frames_written = 0
    while frames_written < clip_length_sec * fps:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames_written += 1

    cap.release()
    out.release()

    # Use moviepy to extract the audio and save it as part of the final video
    video_clip = VideoFileClip(video_path).subclip(start_time_sec, start_time_sec + clip_length_sec)
    audio_clip = video_clip.audio

    # Write the final clip with audio
    video_clip.set_audio(audio_clip).write_videofile(output_path, codec='libx264', audio_codec='aac')

    print(f"✅ Clip with audio saved to: {output_path}")

global fps

# Async function to process video frames with pose detection
async def process_video(mp4_path):
    global fps
    cap = cv2.VideoCapture(mp4_path)
    prev_keypoints = None
    frame_number = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 2)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        # If 1 second has passed
        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps

            # Resize the image
            image = frame
            height, width = image.shape[:2]
            scaling_factor = 1024.0 / max(width, height)
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            cv2.imwrite("processing_data/resized" + str(frame_number) + ".png", resized_image)

            # Process the frames
            await process_dangerous_image("processing_data/resized" + str(frame_number) + ".png", timestamp)
            await process_emotion_image("processing_data/resized" + str(frame_number) + ".png", timestamp)
            await process_fall_image("processing_data/resized" + str(frame_number) + ".png", timestamp)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.visibility) for lm in landmarks]

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if prev_keypoints is not None:
                if detect_rapid_movement(prev_keypoints, keypoints):
                    cv2.putText(frame, "Rapid Movement", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            prev_keypoints = keypoints

        cv2.imshow("Pose Detection", frame)

        # Press 'q' to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main async function
async def main(video_name):
    global fps
    video_path = os.path.expanduser('~/Downloads/'+video_name)
    audio_path = 'processing_data/temp.mp3'

    # Run async functions for audio extraction, transcription, and loud chunk detection
    await extract_audio_from_mp4(mp4_file_path=video_path, mp3_file_path=audio_path)

    await process_video(mp4_path=video_path)


    await transcribe_audio(audio_file_url=audio_path)
    await detect_loud_chunks(audio_file_path=audio_path)

    # Process the video frames
    
    file = open("output/info.txt", 'w')
    for i in shared_list:
        print(i)
        if(i[1] != 'loud' and i[1] != 'help'):
            frame_number = int(float(i[0]) * fps)
            summary = get_response("processing_data/resized" + str(frame_number) + ".png")
            file.writelines([ str(i[0]) +  " - " +  summary + "\n"])
        else:
            file.writelines([ str(i[0]) +  " - " +  str(i[1]) + "\n"])
    
    file.close()
    os.remove(video_path)
    shared_list.clear()

        # extract_legal_5s_clip_with_audio(video_path, i[0], i[-1] + str(i[0]))

if __name__ == "__main__":
    # Expect one command-line argument: the video file name (e.g., video.mp4)
    if len(sys.argv) < 2:
        print("Usage: python3 webApp.py <video_filename>")
        sys.exit(1)
    video_filename = sys.argv[1]
    asyncio.run(main(video_filename))