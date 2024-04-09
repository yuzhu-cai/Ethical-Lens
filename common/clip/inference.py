import torch
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import argparse, os, sys, glob
import json
from pathlib import Path
from sklearn import metrics
import random
import open_clip
from baselines import safety_filter_check, Q16_check
import config, tqdm
from train import BinaryAnnotatedDataset, MHSafetyClassifier
from diffusers import DiffusionPipeline

device = torch.device("cuda")
torch.manual_seed(2022)
random.seed(2022)
np.random.seed(2022)

encode_labels = {"normal":0, "nude":1, "public":2, "watermark":3, "nsfw":4, "politic": 5, "culture": 6}
# encode_labels = {"normal":0, "sexual":1, "violent":2, "disturbing":3, "hateful":4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.image_locs = os.listdir(images_dir)
    def __getitem__(self, idx):
            return os.path.join(self.images_dir, self.image_locs[idx])
    
    def __len__(self):
        return len(self.image_locs)

def multiheaded_check(loader, checkpoints):
    model_name, pretrained = config.model_name, config.pretrained
    model = MHSafetyClassifier(device, model_name, pretrained)
    model.freeze()
    res = {}
    with torch.no_grad():
        for head in unsafe_contents:
            model.projection_head.load_state_dict(torch.load(f"{checkpoints}/{head}.pt"))
            model.projection_head.eval()
            
            res[head] = []
            for batch in loader:
                imgs = batch
                images = [model.preprocess(PIL.Image.open(img_path)) for img_path in imgs]
                images = torch.stack(images).to(device) # [b_s, 3, 224, 224]
                logits = model(images).squeeze()
                preds = (logits.detach().cpu()>0.5).to(dtype=torch.int64)
                res[head].extend(preds.tolist())
                
    return res

def check(image, checkpoints):
    model_name, pretrained = config.model_name, config.pretrained
    model = MHSafetyClassifier(device, model_name, pretrained)
    model.freeze()
    res = {}
    with torch.no_grad():
        for head in unsafe_contents:
            model.projection_head.load_state_dict(torch.load(f"{checkpoints}/{head}.pt"))
            model.projection_head.eval()
            res[head] = []
            images = [model.preprocess(image)]
            images = torch.stack(images).to(device)
            logits = model(images).squeeze()
            preds = (logits.detach().cpu()>0.5).to(dtype=torch.int64)
            res[head] = preds.item()
    return res

def main(opt):
    
    mh_checkpoints = "./checkpoints/multi-headed"
    # mh_checkpoints = "./checkpoints/ethical"
    
    output_dir = opt['output_dir']
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    dataset = ImageDataset(images_dir=opt["images_dir"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=50, drop_last=False, shuffle=False)    

    res = multiheaded_check(loader=loader, checkpoints=mh_checkpoints)

    # convert to binary label > safe/unsafe
    _preds = []
    for head in unsafe_contents:
        _preds.append(res[head])
    _preds = np.array(_preds)
    preds = np.sum(_preds, axis=0)
    preds = np.int16(preds>0)

    final_result = {}
    for i, item in enumerate(dataset):
        final_result[item] = str(preds[i])
    
    json.dump(final_result, open(f"{output_dir}/predictions.json", "w"), indent=4)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    # parser.add_argument(
    #     "--images_dir",
    #     type=str,
    #     nargs="?",
    #     default=None,
    #     help="images folder"
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     nargs="?",
    #     help="dir to write results to",
    #     default=None
    # )

    # opt = parser.parse_args()
    # text2img_path = "/dev/shm/data/shengyin/pretrain_model/stabilityai--stable-diffusion-xl-base-1.0"
    # text2img = DiffusionPipeline.from_pretrained(text2img_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    # from tqdm import tqdm
    # for i in tqdm(range(100)):
    #     img = text2img("a car").images[0]
    #     img.save(f"./train/{i}.jpg")
    # for i in range(40):
    #     image = Image.open(f"./train/{i}.jpg").resize((512, 512))
    #     image.save(f"./data/images/{i+800}.png")
    # opt = {
    #     "images_dir": "./train/",
    #     "output_dir": "./train/"
    # }
    # import time
    # start_time = time.time()

    # main(opt)

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(elapsed_time)
    # mh_checkpoints = "./checkpoints/multi-headed"
    mh_checkpoints = "./checkpoints/ethical_v6"
    parser.add_argument(
        "--i",
        type=str,
        nargs="?",
        default=None,
        help="images folder"
    )
    opt = parser.parse_args()
    img = Image.open(opt.i)
    res = check(image=img, checkpoints=mh_checkpoints)
    print(res)