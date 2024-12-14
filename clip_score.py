import torch
import clip
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

def compute_video_clip_score(video_path: str, text_prompt: str) -> Tuple[float, List[float]]:
    """
    Computes average CLIP score between video frames and a text prompt (averaged across video frames).
    
    Args:
        video_path (str): Path to the MP4 video file
        text_prompt (str): Text prompt to compare against frames
    
    Returns:
        Tuple[float, List[float]]: (average_score, list of individual frame scores)
    """
    # Load CLIP model and preprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Encode text prompt
    text = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB and then to PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        
        # Preprocess and encode frame
        frame = preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            frame_features = model.encode_image(frame)
            frame_features /= frame_features.norm(dim=-1, keepdim=True)
            
        # Compute similarity score
        similarity = (100.0 * frame_features @ text_features.T).item()
        frame_scores.append(similarity)
    
    cap.release()
    
    # Compute average score
    average_score = np.mean(frame_scores)
    
    return average_score, frame_scores


def compute_interframe_clip_score(video_path: str) -> Tuple[float, List[float]]:
    """
    Computes average CLIP score between successive video frames (averaged across |F| -1 frame pairs).
    
    Args:
        video_path (str): Path to the MP4 video file
    
    Returns:
        Tuple[float, List[float]]: (average_score, list of individual frame scores)
    """
    # Load CLIP model and preprocessing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    scores = []
    prev_frame = None
    
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        if not prev_frame:
            prev_frame = curr_frame
            continue
            
        # Convert Previous Frame BGR to RGB and then to PIL Image
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        prev_frame = Image.fromarray(prev_frame)

        # Convert Current Frame BGR to RGB and then to PIL Image
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        curr_frame = Image.fromarray(curr_frame)
        
        # Preprocess and encode frames
        prev_frame = preprocess(prev_frame).unsqueeze(0).to(device)
        curr_frame = preprocess(curr_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            curr_frame_features = model.encode_image(curr_frame)
            prev_frame_features = model.encode_image(prev_frame)
            curr_frame_features /= curr_frame_features.norm(dim=-1, keepdim=Tprev
            prev_frame_features /= prev_frame_features.norm(dim=-1, keepdim=True)
            
        # Compute similarity score
        similarity = (100.0 * curr_frame_features @ prev_frame_features.T).item()
        scores.append(similarity)
    
    cap.release()
    
    # Compute average score
    average_score = np.mean(scores)
    
    return average_score, scores

def main_interframe():
    """Example usage of the CLIP score computation"""

    CURR_DIR = os.getcwd()
    TASK = ''
    video_dir = os.path.join(CURR_DIR, TASK)
    clip_scores = []

    for video in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video)
        interframe_clip_score, _ = compute_interframe_clip_score(video_path)
        clip_scores.append(interframe_clip_score)

    mean_clip_score = np.mean(clip_scores)
    print(f'Task: {TASK}')
    print(f"Average CLIP score: {mean_clip_score:.2f}")
    print(f"Number of frames processed: {len(frame_scores)}")
    print(f"Min frame score: {min(frame_scores):.2f}")
    print(f"Max frame score: {max(frame_scores):.2f}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main_interframe()