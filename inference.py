import argparse
import torch
from model import Model
import os

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process videos with pix2pix using text prompts.")
    parser.add_argument('--prompts_file', type=str, required=True, help="Path to the text file containing prompts.")
    parser.add_argument('--output_dir', type=str, default="./outputs", help="Directory to save output videos.")
    args = parser.parse_args()
    
    # Load the model
    model = Model(device="cuda", dtype=torch.float16)

    # Read prompts from the file
    try:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]  # Filter out empty lines
    except FileNotFoundError:
        print(f"Error: Prompts file '{args.prompts_file}' not found.")
        return

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 40}

    # Process the video for each prompt
    for prompt in prompts:
        sanitized_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:20]  # Make the filename safe
        output_path, fps = os.path.join(args.output_dir, f"video_instruct_pix2pix_{sanitized_prompt}.mp4"), 4
        print(f"Processing with prompt: '{prompt}'")
        model.process_text2video(prompt, fps = fps, path = output_path, **params)

if __name__ == "__main__":
    main()
