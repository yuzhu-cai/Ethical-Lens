import torch 
import random
import argparse

from PIL import Image

from model.ethicallens import Ethicallens
from common.random_seed import prepare_seed

def set_cfg(args):
    cfg = {
        'text2img_path' : args.text2img_path,
        'resolution' : args.resolution,
        'relative_gpu_index' : args.relative_gpu_index,

        'faceedit_device': f'cuda:{args.faceedit_device}',
        'fairface' : {
            'relative_gpu_index' : f"{args.faceedit_device}",
            'SAVE_DETECTED_AT': "results/detected_faces_generate"
        },
        'if_fairface' : args.if_fairface,

        'checkpoints': args.checkpoints,
        'clipseg_path': args.clipseg_path,
        'llm_path': args.llm_path,
        'peft_path': args.peft_path
    }
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--relative_gpu_index", type=int, default=0)
    parser.add_argument("--faceedit_device", type=int, default=1)
    parser.add_argument("--if_fairface", type=bool, default=True)
    parser.add_argument("--checkpoints", type=str, default='./checkpoints')
    parser.add_argument("--clipseg_path", type=str, default='/home/ubuntu/DATA1/yuzhucai/prestrain_model/CIDAS--clipseg-rd64-refined')
    parser.add_argument("--llm_path", type=str, default="/home/ubuntu/yuxiwei/llm_finetune/Qwen-7B-Chat")
    parser.add_argument("--peft_path", type=str, default="/home/ubuntu/yuxiwei/Qwen/output/merged_all_0223")
    parser.add_argument("--text2img_path", type=str, default="/dev/shm/data/shengyin/pretrain_model/models--dreamlike-art--dreamlike-diffusion-1.0/snapshots/9fb5a6463bf79d81152e715e8d2a8b988f96c790")

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--state", type=str, default='Yes')


    args = parser.parse_args()

    cfg = set_cfg(args)
    
    ethicallens = Ethicallens(cfg=cfg)

    random.seed()
    random_seed = random.randint(0, 1000000)
    prepare_seed(random_seed)
    data = {
            'seed': random_seed,
            'status': 'SUCSESS',
        }
    ethicallens.generate(
        prompt=args.prompt,
        img_number=args.num_images_per_prompt,
        data=data,
        state=args.state
    )
    for i, img in enumerate(data['images']):
        img.save(f'./results/imgs/{i}.jpg')
