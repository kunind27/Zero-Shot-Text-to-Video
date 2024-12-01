import torch
from model import Model

model = Model(device="cuda", dtype=torch.float16)

prompt = 'make it Van Gogh Starry Night'
video_path = '__assets__/pix2pix video/camel.mp4'
out_path = f'./video_instruct_pix2pix_{prompt}.mp4'
model.process_pix2pix(video_path, prompt=prompt, save_path=out_path)
