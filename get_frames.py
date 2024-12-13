import cv2
import os

def process_videos_in_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        video_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(video_path):
            continue  # Skip directories or invalid files

        # Check if the file is a video (based on extension)
        if not file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            print(f"Skipping non-video file: {file_name}")
            continue

        # Create a unique folder for this video
        video_name, _ = os.path.splitext(file_name)
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        # Process the video
        print(f"Processing video: {file_name}")
        extract_frames(video_path, video_output_folder, video_name)

def extract_frames(video_path, output_folder, video_name):
    video_capture = cv2.VideoCapture(video_path)
    frame_index = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break  # No more frames to read

        # Save each frame
        frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_index:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_index += 1

    video_capture.release()
    print(f"Finished extracting frames from {os.path.basename(video_path)}. Frames saved in {output_folder}")

# Example usage
input_folder = "/Users/manavdoshi/Zero-Shot-Text-to-Video/videos_2_seconds/background_changes"  # Replace with your videos folder
output_folder = "/Users/manavdoshi/Zero-Shot-Text-to-Video/videos_2_seconds/background_changes"  # Replace with your desired output folder
process_videos_in_folder(input_folder, output_folder)
