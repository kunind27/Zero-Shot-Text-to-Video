import torch
from model import Model

model = Model(device = "cuda:0", dtype = torch.float16)

prompt_list = ["A cat is running on the grass",
               "A panda is playing guitar on times square",
               "A man is running in the snow",
               "An astronaut is skiing down the hill",
               "A panda surfing on a wakeboard",
               "A bear dancing on times square",
               "A man is riding a bicycle in the sunshine",
               "A horse galloping on a street",
               "A tiger walking alone down the street",
               "A panda surfing on a wakeboard",
               "A horse galloping on a street",
               "A cute cat running in a beautiful meadow",
               "A panda walking alone down the street",
               "A dog is walking down the street",
               "An astronaut is waving his hands on the moon",
               "A panda dancing on times square",
               "A bear walking on a mountain",
               "A gorilla walking down the street",
               "A man is walking in the rain"]

prompt = "A horse galloping on a street"
params = {"t0": 44, "t1": 47 , "motion_field_strength_x" : 12, "motion_field_strength_y" : 12, "video_length": 8}

for prompt in prompt_list:
    out_path, fps = f"./text2video_{prompt.replace(' ','_')}.mp4", 4
    model.process_text2video(prompt, fps = fps, path = out_path, **params)