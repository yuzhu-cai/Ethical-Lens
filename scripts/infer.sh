gpu=2,3


# text2img_path="dreamlike-art/dreamlike-diffusion-1.0"
text2img_path="" # your path to Text2Image Model
llm_path=""  # your path to Text Scrutiny LLM
clipseg_path="" # yout path to CIDAS/clipseg-rd64-refined
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
    --clipseg_path $clipseg_path  \
    --checkpoints $checkpoints  \
    --resolution $resolution  \
    --relative_gpu_index $relative_gpu_index  \
    --faceedit_device $faceedit_device  \
    --if_fairface $if_fairface  \
    --prompt "$prompt"  \
    --num_images_per_prompt $num_images_per_prompt  \
    --state $state
