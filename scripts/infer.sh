gpu=2,3


# text2img_path="dreamlike-art/dreamlike-diffusion-1.0"
text2img_path="/dev/shm/data/shengyin/pretrain_model/models--dreamlike-art--dreamlike-diffusion-1.0/snapshots/9fb5a6463bf79d81152e715e8d2a8b988f96c790"
llm_path="/home/ubuntu/yuxiwei/llm_finetune/Qwen-7B-Chat"
peft_path="/home/ubuntu/yuxiwei/Qwen/output/merged_all_0223"
clipseg_path="/home/ubuntu/DATA1/yuzhucai/prestrain_model/CIDAS--clipseg-rd64-refined"
checkpoints="./checkpoints"

resolution=512
relative_gpu_index=0
faceedit_device=1
if_fairface=True

prompt="a doctor"
num_images_per_prompt=4
state="Yes"

CUDA_VISIBLE_DEVICES=$gpu python infer.py \
    --text2img_path $text2img_path  \
    --llm_path $llm_path  \
    --peft_path $peft_path  \
    --clipseg_path $clipseg_path  \
    --checkpoints $checkpoints  \
    --resolution $resolution  \
    --relative_gpu_index $relative_gpu_index  \
    --faceedit_device $faceedit_device  \
    --if_fairface $if_fairface  \
    --prompt "$prompt"  \
    --num_images_per_prompt $num_images_per_prompt  \
    --state $state
