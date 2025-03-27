#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-scavenger
#SBATCH --mem=128gb  
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --partition=vulcan-scavenger
#SBATCH --output=/vulcanscratch/yiranx/codes/diffusers/jobs/slurm_output-%j.out

# export MODEL_NAME="black-forest-labs/FLUX.1-dev"
# export DATASET_NAME="/fs/vulcan-projects/sc4d/datasets/garden"
# export OUTPUT_DIR="/fs/vulcan-projects/sc4d/flux_ckpts/garden-view-Flux-LoRA"

# accelerate launch examples/advanced_diffusion_training/train_dreambooth_lora_flux_advanced.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --instance_prompt="photo of a garden view with TOK" \
#   --output_dir=$OUTPUT_DIR \
#   --caption_column="prompt" \
#   --mixed_precision="bf16" \
#   --resolution=1024 \
#   --train_batch_size=1 \
#   --repeats=1 \
#   --report_to="wandb"\
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --learning_rate=1.0 \
#   --text_encoder_lr=1.0 \
#   --optimizer="prodigy"\
#   --train_text_encoder_ti\
#   --train_text_encoder_ti_frac=0.5\
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --rank=8 \
#   --max_train_steps=2000 \
#   --checkpointing_steps=2000 \
#   --seed="0" \
#   --push_to_hub

python train_flux_dual_adapter.py   --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --output_dir="/vulcanscratch/yiranx/sc4d/flux_ckpts/flux-ip-adapter-garden-16" \
  --train_batch_size=2 \
  --num_train_epochs=500 \
  --mixed_precision="bf16" \
  --save_steps=100 \
  --rgb_adapter_scale=0.7 \
  --feature_adapter_scale=0.3 \
  --resolution=224 \
  --image_root_path /fs/vulcan-projects/sc4d/datasets/garden_16/input/ \
  --feature_root_path /fs/vulcan-projects/sc4d/datasets/garden_16/plucker.npy
