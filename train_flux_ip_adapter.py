#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import json
import itertools
import logging
import math
import os
import random
import time
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import save_file
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.models.embeddings import ImageProjection
from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
from diffusers.utils import check_min_version
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory

# Will error if the minimal version of diffusers is not installed.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


class MyDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenized prompts.
    """
    def __init__(
        self, 
        json_file, 
        tokenizer, 
        image_encoder,
        image_processor,
        size=512,
        t_drop_rate=0.05, 
        i_drop_rate=0.05, 
        ti_drop_rate=0.05, 
        image_root_path=""
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        raw_image = exif_transpose(raw_image)
        if not raw_image.mode == "RGB":
            raw_image = raw_image.convert("RGB")
            
        image = self.transform(raw_image)
        
        # Process image for CLIP
        clip_image = self.image_processor(images=raw_image, return_tensors="pt").pixel_values.squeeze(0)

        # Conditioning dropout 
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        
        # Tokenize text
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
        }

    def __len__(self):
        return len(self.data)


class DummyDataset(torch.utils.data.Dataset):
    """
    A dataset that generates random dummy data on-the-fly for testing the training pipeline.
    """
    def __init__(
        self, 
        tokenizer, 
        image_processor,
        size=224,
        length=1000,
        t_drop_rate=0.05, 
        i_drop_rate=0.05, 
        ti_drop_rate=0.05,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.size = size
        self.length = length
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        
        # Words for random caption generation
        self.nouns = ["dog", "cat", "person", "tree", "house", "car", "mountain", "river", "city", "sunset", 
                  "building", "flower", "bird", "beach", "forest", "sky", "computer", "phone", "book"]
        
        self.adjectives = ["happy", "sad", "bright", "dark", "colorful", "big", "small", "beautiful", "strange",
                      "tall", "tiny", "red", "green", "blue", "yellow", "old", "new", "shiny", "wooden"]
        
        self.verbs = ["running", "sitting", "standing", "flying", "sleeping", "jumping", "walking", "swimming",
                 "eating", "drinking", "watching", "reading", "writing", "playing", "working"]
        
        self.prepositions = ["in", "on", "at", "by", "with", "under", "over", "behind", "in front of", "next to"]
        
        # Create normalization transforms
        self.transform = transforms.Normalize([0.5], [0.5])

    def __len__(self):
        return self.length
    
    def generate_random_image(self):
        """Generate a random tensor image."""
        # Create a random RGB image tensor directly - skipping PIL for efficiency
        # Values between 0 and 1 before normalization
        random_tensor = torch.rand(3, self.size, self.size)
        return self.transform(random_tensor)
    
    def generate_random_clip_image(self):
        """Generate a random tensor formatted like CLIP processor output."""
        # Create random values in the expected CLIP normalization range
        # CLIP normalization: images = (images - mean) / std
        # where mean=[0.48145466, 0.4578275, 0.40821073] and std=[0.26862954, 0.26130258, 0.27577711]
        # We'll skip the preprocessing and generate values directly in expected range
        return torch.rand(3, self.size, self.size)
    
    def generate_random_text(self, min_words=3, max_words=12):
        """Generate a random caption."""
        num_words = random.randint(min_words, max_words)
        words = []
        
        # Start with an article and adjective
        words.append(random.choice(["A", "The"]))
        words.append(random.choice(self.adjectives))
        words.append(random.choice(self.nouns))
        
        # Add random phrases until we reach desired length
        while len(words) < num_words:
            phrase_type = random.randint(0, 2)
            if phrase_type == 0:
                words.append(random.choice(self.verbs))
            elif phrase_type == 1:
                words.append(random.choice(self.prepositions))
                words.append(random.choice(["a", "the"]))
                if random.random() > 0.5:
                    words.append(random.choice(self.adjectives))
                words.append(random.choice(self.nouns))
            else:
                words.append(random.choice(self.adjectives))
        
        caption = " ".join(words) + "."
        return caption
    
    def __getitem__(self, idx):
        # Generate random image tensors - one for VAE input, one for CLIP
        image = self.generate_random_image()
        clip_image = self.generate_random_clip_image()
        
        # Generate random text
        text = self.generate_random_text()
        
        # Conditioning dropout
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        
        # Tokenize text
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
        }


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.stack([example["clip_image"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
    }


class FluxIPAdapter(torch.nn.Module):
    """IP-Adapter for Flux model"""

    def __init__(self, transformer, image_proj_model, num_tokens=4):
        super().__init__()
        self.transformer = transformer
        self.image_proj_model = image_proj_model
        self.num_tokens = num_tokens
        
        # Setup IP adapter processor for transformer attention blocks
        attn_procs = {}
        for name in transformer.attn_processors.keys():
            if name.startswith("single_transformer_blocks"):
                attn_processor_class = transformer.attn_processors[name].__class__
                attn_procs[name] = attn_processor_class()
            else:
                attn_procs[name] = FluxIPAdapterJointAttnProcessor2_0(
                    # hidden_size=transformer.config.out_channels,
                    hidden_size=3072,  # hardcoded for now
                    cross_attention_dim=transformer.config.joint_attention_dim,
                    scale=1.0,
                    num_tokens=num_tokens,
                )
        
        self.transformer.set_attn_processor(attn_procs)
        self.transformer.config.encoder_hid_dim_type = "ip_image_proj"

    def forward(self, hidden_states, timesteps, encoder_hidden_states, pooled_prompt_embeds, image_embeds, text_ids, img_ids, guidance=None):
        # Project image embeddings to adapter tokens
        ip_tokens = self.image_proj_model(image_embeds)
        _joint_attention_kwargs = {'ip_adapter_image_embeds': ip_tokens}
        # Forward through transformer
        noise_pred = self.transformer(
            hidden_states=hidden_states,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=_joint_attention_kwargs,
            guidance=guidance,
            # hidden_states=latents,
            # timestep=timestep / 1000,
            # guidance=guidance,
            # pooled_projections=pooled_prompt_embeds,
            # encoder_hidden_states=prompt_embeds,
            # txt_ids=text_ids,
            # img_ids=latent_image_ids,
            # joint_attention_kwargs=self.joint_attention_kwargs,
            # return_dict=False,
        ).sample
        
        return noise_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Flux IP-Adapter")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained Flux model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier.",
    )
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        help="Use randomly generated dummy data instead of real dataset.",
    )
    parser.add_argument(
        "--dummy_data_size",
        type=int,
        default=1000,
        help="Number of dummy data samples to generate (only used with --use_dummy_data).",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        help="Path to JSON file containing training data (image paths and captions). Not required if using dummy data.",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="",
        help="Root path for images referenced in the JSON file. Not required if using dummy data.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Path to CLIP vision model for IP-Adapter.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-ip-adapter",
        help="The output directory where the model weights will be saved.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=4,
        help="Number of IP adapter tokens to use.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum T5 sequence length to use",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=8, 
        help="Batch size for training."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="CFG scale for training (if the model supports it).",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers to use for dataloader.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints. Deletes the older checkpoints.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache latents to save computational resources.",
    )
    parser.add_argument(
        "--t_drop_rate", 
        type=float, 
        default=0.05,
        help="Text dropout rate for training.",
    )
    parser.add_argument(
        "--i_drop_rate", 
        type=float, 
        default=0.05,
        help="Image dropout rate for training.",
    )
    parser.add_argument(
        "--ti_drop_rate", 
        type=float, 
        default=0.05,
        help="Text-Image dropout rate for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=500, 
        help="Number of steps for the learning rate warmup."
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help='We default to the "none" weighting scheme for uniform sampling and uniform loss',
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing or not.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to.',
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Validate arguments
    if not args.use_dummy_data and args.data_json_file is None:
        raise ValueError("Must either use --use_dummy_data or provide --data_json_file")

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        
    if args.use_dummy_data:
        print(f"\n===== USING DUMMY DATA =====")
        print(f"This run will use randomly generated data for training")
        print(f"Number of samples: {args.dummy_data_size}")
        print(f"This is intended for testing the training pipeline, not for actual model training")
        print(f"==============================\n")

    # Load models
    # Load scheduler and tokenizer
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )
    
    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_two = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )
    
    # Load transformer and VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    
    # Load CLIP image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    image_processor = CLIPImageProcessor.from_pretrained(args.image_encoder_path)
    
    # Create IP Adapter projection model
    image_proj_model = ImageProjection(
        cross_attention_dim=transformer.config.joint_attention_dim,
        image_embed_dim=image_encoder.config.projection_dim,
        num_image_text_embeds=args.num_tokens,
    )
    
    # Freeze all models except for IP adapter
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    transformer.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Create IP adapter
    ip_adapter = FluxIPAdapter(transformer, image_proj_model, num_tokens=args.num_tokens)

    # Setup weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Enable gradient checkpointing for transformer if requested
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Create optimizer
    params_to_optimize = list(ip_adapter.image_proj_model.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create dataset and dataloader
    if args.use_dummy_data:
        logger.info(f"Using randomly generated dummy data with {args.dummy_data_size} samples")
        train_dataset = DummyDataset(
            tokenizer=tokenizer_one,
            image_processor=image_processor,
            size=args.resolution,
            length=args.dummy_data_size,
            t_drop_rate=args.t_drop_rate,
            i_drop_rate=args.i_drop_rate,
            ti_drop_rate=args.ti_drop_rate,
        )
    else:
        if args.data_json_file is None:
            raise ValueError("--data_json_file must be provided when not using dummy data")
        
        logger.info(f"Loading dataset from {args.data_json_file}")
        train_dataset = MyDataset(
            args.data_json_file,
            tokenizer=tokenizer_one,
            image_encoder=image_encoder,
            image_processor=image_processor,
            size=args.resolution,
            t_drop_rate=args.t_drop_rate,
            i_drop_rate=args.i_drop_rate,
            ti_drop_rate=args.ti_drop_rate,
            image_root_path=args.image_root_path,
        )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare accelerator
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(
        ip_adapter, optimizer, train_dataloader
    )

    # Cache latents if requested
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels)) 
    vae_config_shift_factor = getattr(vae.config, "shift_factor", 0.0)
    vae_config_scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
    
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["images"] = batch["images"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                latents = vae.encode(batch["images"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents_cache.append(latents)

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Prepare everything
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Print training info
    logger.info("***** Running IP-Adapter Training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Setup progress bar and tracking
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device=None):
        """Helper function to get sigmas for the FlowMatchEulerDiscreteScheduler."""
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Training loop
    for epoch in range(args.num_train_epochs):
        ip_adapter.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ip_adapter):
                # Convert images to latents
                if args.cache_latents:
                    latents = latents_cache[step]
                else:
                    with torch.no_grad():
                        batch["images"] = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        latents = vae.encode(batch["images"]).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                
                # Prepare latents
                latents = (latents - vae_config_shift_factor) * vae_config_scaling_factor
                latents = latents.to(dtype=weight_dtype)
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
                
                # Add noise according to flow matching
                sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype, device=latents.device)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                
                # Pack latents for transformer
                packed_noisy_latents = FluxPipeline._pack_latents(
                    noisy_latents,
                    batch_size=latents.shape[0],
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[2],
                    width=latents.shape[3],
                )
                
                # Create image IDs
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    latents.shape[0],
                    latents.shape[2] // 2,
                    latents.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                # Process CLIP image embeddings
                with torch.no_grad():
                    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                
                # Apply conditional dropout
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
                
                # Process text embeddings
                with torch.no_grad():
                    # CLIP text embeddings
                    text_embeds_one = text_encoder_one(batch["text_input_ids"].to(accelerator.device))[0]
                    
                    # T5 text embeddings
                    text_embeds_two = text_encoder_two(
                        input_ids=tokenizer_two(
                            [tokenizer_one.decode(ids) for ids in batch["text_input_ids"]],
                            padding="max_length",
                            max_length=args.max_sequence_length,
                            truncation=True,
                            return_tensors="pt",
                        ).input_ids.to(accelerator.device)
                    )[0]
                    
                    # Calculate pooled text embeddings
                    pooled_text_embeddings = text_encoder_one(
                        batch["text_input_ids"].to(accelerator.device), output_hidden_states=True
                    ).pooler_output
                
                # Setup for transformer forward pass
                text_ids = torch.zeros(bsz, text_embeds_two.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
                
                # Handle guidance if supported
                if hasattr(transformer.config, "guidance_embeds") and transformer.config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None
                
                # Forward pass through IP adapter
                noise_pred = ip_adapter(
                    hidden_states=packed_noisy_latents,
                    timesteps=timesteps / 1000,  # Normalize timesteps for Flux model
                    encoder_hidden_states=text_embeds_two,
                    pooled_prompt_embeds=pooled_text_embeddings,
                    image_embeds=image_embeds,
                    text_ids=text_ids,
                    img_ids=latent_image_ids,
                    guidance=guidance,
                )
                
                # Unpack predictions
                noise_pred = FluxPipeline._unpack_latents(
                    noise_pred,
                    height=latents.shape[2] * vae_scale_factor,
                    width=latents.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                
                # Calculate loss weighting
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                
                # Flow matching loss
                target = noise - latents
                
                # Calculate loss
                loss = torch.mean(
                    (weighting.float() * (noise_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                
                # Backpropagate
                accelerator.backward(loss)
                
                # Update parameters
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Track progress
                progress_bar.update(1)
                global_step += 1
                total_loss += loss.detach().item()
                
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                
                # Save checkpoint
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        # Save IP-Adapter weights
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Get model state dict
                        ip_adapter_state_dict = accelerator.unwrap_model(ip_adapter.image_proj_model).state_dict()
                        
                        # Save as safetensors
                        save_file(ip_adapter_state_dict, os.path.join(save_path, "ip_adapter_projection.safetensors"))
                        
                        logger.info(f"Saved checkpoint to {save_path}")
                        
                        # Handle checkpoint cleanup
                        if args.save_total_limit is not None:
                            checkpoints = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            # If we exceed the number of checkpoints, remove the oldest
                            if len(checkpoints) > args.save_total_limit:
                                num_to_remove = len(checkpoints) - args.save_total_limit
                                removing_checkpoints = checkpoints[:num_to_remove]
                                
                                logger.info(f"Removing old checkpoints: {', '.join(removing_checkpoints)}")
                                for checkpoint in removing_checkpoints:
                                    path = os.path.join(args.output_dir, checkpoint)
                                    if os.path.exists(path):
                                        shutil.rmtree(path)
                
                # Check if we've reached max_train_steps
                if global_step >= args.max_train_steps:
                    break
        
        # Log per-epoch statistics
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} completed with average loss: {avg_loss:.4f}")
    
    # Final save
    if accelerator.is_main_process:
        # Save IP-Adapter weights
        save_path = os.path.join(args.output_dir, "ip_adapter_final")
        os.makedirs(save_path, exist_ok=True)
        
        # Get model state dict
        ip_adapter_state_dict = accelerator.unwrap_model(ip_adapter.image_proj_model).state_dict()
        
        # Save as safetensors
        save_file(ip_adapter_state_dict, os.path.join(save_path, "ip_adapter_projection.safetensors"))
        
        logger.info(f"Saved final IP-Adapter weights to {save_path}")
        
    # Clean up
    accelerator.end_training()


if __name__ == "__main__":
    main()
