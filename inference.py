import torch
from models.pipeline import VchitectXLPipeline
import random
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def infer(args):
    pipe = VchitectXLPipeline(args.ckpt_path)
    idx = 0

    with open(args.test_file,'r') as f:
        for lines in f.readlines():
            for seed in range(5):
                set_seed(seed)
                prompt = lines.strip('\n')
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    video = pipe(
                        prompt,
                        negative_prompt="",
                        num_inference_steps=100,
                        guidance_scale=7.5,
                        width=768,
                        height=432, #480x288  624x352 432x240 768x432
                        frames=40
                    )

                images = video

                from utils import save_as_mp4
                import sys,os
                duration = 1000 / 8

                save_dir = args.save_dir
                os.makedirs(save_dir,exist_ok=True)

                idx += 1
                
                save_as_mp4(images, os.path.join(save_dir, f"sample_{idx}_seed{seed}")+'.mp4', duration=duration)
                
import sys,os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_known_args()[0]
    infer(args)

if __name__ == "__main__":
    main()
