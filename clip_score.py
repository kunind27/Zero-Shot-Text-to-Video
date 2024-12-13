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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Preprocess and encode frame
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        # Compute similarity score
        similarity = (100.0 * image_features @ text_features.T).item()
        frame_scores.append(similarity)
    
    cap.release()
    
    # Compute average score
    average_score = np.mean(frame_scores)
    
    return average_score, frame_scores

def main():
    """Example usage of the CLIP score computation"""
    video_path = "example.mp4"
    text_prompt = "a person walking on the beach"
    
    try:
        avg_score, frame_scores = compute_video_clip_score(video_path, text_prompt)
        print(f"Average CLIP score: {avg_score:.2f}")
        print(f"Number of frames processed: {len(frame_scores)}")
        print(f"Min frame score: {min(frame_scores):.2f}")
        print(f"Max frame score: {max(frame_scores):.2f}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()