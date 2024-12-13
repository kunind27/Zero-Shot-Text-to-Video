import os
import torch
from model import Model

model = Model(device = "cuda:1", dtype = torch.float16)

instructions = [('make it Van Gogh Starry Night style', 'camel.mp4'),
                ('make it Expressionism style', 'camel.mp4'),
                ('make it snowy', 'mini-cooper.mp4'),
                ('make it Picasso style', 'mini-cooper.mp4'),
                ('make it Van Gogh Starry Night style', 'white-swan.mp4'),
                ('replace swan with mallard', 'white-swan.mp4')]

VIDEO_DIR = './__assets__/pix2pixvideo/'

for prompt, video in instructions:
    video_path = os.path.join(VIDEO_DIR, video)
    out_path = f'../data/videos/pix2pix/video_instruct_pix2pix_{prompt}_{video}'
    model.process_pix2pix(video_path, prompt=prompt, save_path=out_path)