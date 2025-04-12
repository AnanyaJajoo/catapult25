import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from a video at a specified interval and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        frame_interval (int): Interval between frames to extract (e.g., every nth frame).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save every nth frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete. {saved_count} frames saved to {output_dir}.")

if __name__ == "__main__":
    # Example usage
    video_path = "path_to_your_video.mp4"  # Replace with your video file path
    output_dir = "output_frames"  # Replace with your desired output directory
    frame_interval = 30  # Adjust the interval as needed (e.g., every 30th frame)

    extract_frames(video_path, output_dir, frame_interval)