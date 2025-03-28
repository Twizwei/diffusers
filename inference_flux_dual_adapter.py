#!/usr/bin/env python
# coding=utf-8

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
from diffusers.models.embeddings import ImageProjection, MultiIPAdapterImageProjection
from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from safetensors.torch import load_file


class CustomFeatureEncoder(nn.Module):
    """Simple encoder for feature maps"""
    def __init__(self, in_channels=12, hidden_dim=256, out_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),  # Resize to fixed spatial dimensions
            nn.Flatten(),
            nn.Linear(hidden_dim * 16 * 16, out_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained Flux Dual IP-Adapter")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained Flux model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained dual adapter checkpoint directory.",
    )
    parser.add_argument(
        "--rgb_image_path",
        type=str,
        required=True,
        help="Path to the input RGB image for conditioning.",
    )
    parser.add_argument(
        "--feature_map_path",
        type=str,
        default=None,
        help="Path to the input feature map for conditioning (optional, will generate random if not provided).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative text prompt for image generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output image.",
    )
    parser.add_argument(
        "--rgb_adapter_scale",
        type=float,
        default=None,
        help="Scale for the RGB adapter's influence. If None, uses the value from config.",
    )
    parser.add_argument(
        "--feature_adapter_scale",
        type=float,
        default=None,
        help="Scale for the feature adapter's influence. If None, uses the value from config.",
    )
    parser.add_argument(
        "--feature_channels",
        type=int,
        default=12,
        help="Number of channels in the feature maps.",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=72,
        help="Size of feature maps (height and width).",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="CFG scale for text guidance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--refenrence_id",
        type=int,
        default=0,
        help="Reference ID for the feature map.",
    )
    parser.add_argument(
        "--tgt_id",
        type=int,
        default=1,
        help="Target ID for the feature map.",
    )
    
    return parser.parse_args()


def load_adapter_config(path):
    """Load adapter configuration from the checkpoint directory."""
    config_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        # Default config if no config file is found
        return {
            "rgb_adapter_scale": 1.0,
            "feature_adapter_scale": 1.0,
            "num_tokens": 4,
            "feature_channels": 12,
            "feature_encoder_hidden_dim": 256,
            "feature_encoder_output_dim": 768,
        }


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dtype = torch.float16 if device == "cuda" else torch.float32
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Load adapter config
    adapter_config = load_adapter_config(args.adapter_path)
    
    # Use command line args if provided, otherwise use config values
    num_tokens = adapter_config.get("num_tokens", 4)
    feature_channels = args.feature_channels or adapter_config.get("feature_channels", 12)
    feature_encoder_hidden_dim = adapter_config.get("feature_encoder_hidden_dim", 256)
    feature_encoder_output_dim = adapter_config.get("feature_encoder_output_dim", 768)
    rgb_adapter_scale = args.rgb_adapter_scale or adapter_config.get("rgb_adapter_scale", 0.7)
    feature_adapter_scale = args.feature_adapter_scale or adapter_config.get("feature_adapter_scale", 0.3)
    
    # Initialize image encoder and feature extractor
    print("Loading models...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=dtype
    )
    feature_extractor = CLIPImageProcessor()
    
    # Create the base pipeline
    pipe = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor
    ).to(device)
    
    # Create feature encoder
    feature_encoder = CustomFeatureEncoder(
        in_channels=feature_channels,
        hidden_dim=feature_encoder_hidden_dim,
        out_dim=feature_encoder_output_dim
    ).to(device).to(dtype)
    
    # Load IP-Adapter components
    rgb_adapter = ImageProjection(
        cross_attention_dim=pipe.transformer.config.joint_attention_dim,
        image_embed_dim=image_encoder.config.projection_dim,
        num_image_text_embeds=num_tokens
    ).to(device).to(dtype)
    
    feature_adapter = ImageProjection(
        cross_attention_dim=pipe.transformer.config.joint_attention_dim,
        image_embed_dim=feature_encoder_output_dim,
        num_image_text_embeds=num_tokens
    ).to(device).to(dtype)
    
    # Combine adapters with MultiIPAdapterImageProjection
    multi_adapter = MultiIPAdapterImageProjection([rgb_adapter, feature_adapter]).to(device).to(dtype)
    
    # Load trained weights
    print(f"Loading weights from {args.adapter_path}")
    
    # Try to load weights from final or checkpoint directory
    rgb_adapter_path = os.path.join(args.adapter_path, "rgb_adapter.safetensors")
    feature_adapter_path = os.path.join(args.adapter_path, "feature_adapter.safetensors")
    feature_encoder_path = os.path.join(args.adapter_path, "feature_encoder.safetensors")
    
    if not os.path.exists(rgb_adapter_path):
        print(f"Warning: Could not find {rgb_adapter_path}. Looking for alternatives...")
        # Try looking in most recent checkpoint
        checkpoint_dirs = [d for d in os.listdir(args.adapter_path) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
            rgb_adapter_path = os.path.join(args.adapter_path, latest_checkpoint, "rgb_adapter.safetensors")
            feature_adapter_path = os.path.join(args.adapter_path, latest_checkpoint, "feature_adapter.safetensors")
            feature_encoder_path = os.path.join(args.adapter_path, latest_checkpoint, "feature_encoder.safetensors")
            print(f"Using weights from {latest_checkpoint}")
    
    # Load weights
    rgb_adapter.load_state_dict(load_file(rgb_adapter_path))
    feature_adapter.load_state_dict(load_file(feature_adapter_path))
    feature_encoder.load_state_dict(load_file(feature_encoder_path))
    
    # Set up attention processors for IP-Adapter
    attn_procs = {}
    for name in pipe.transformer.attn_processors.keys():
        if name.startswith("single_transformer_blocks"):
            attn_processor_class = pipe.transformer.attn_processors[name].__class__
            attn_procs[name] = attn_processor_class()
        else:
            attn_procs[name] = FluxIPAdapterJointAttnProcessor2_0(
                hidden_size=3072,  # hardcoded for now
                cross_attention_dim=pipe.transformer.config.joint_attention_dim,
                scale=1.0,
                num_tokens=[num_tokens, num_tokens],
                dtype=dtype,
                device=device
            )
    
    # Set up the pipeline for IP-Adapter
    pipe.transformer.encoder_hid_proj = multi_adapter
    pipe.transformer.set_attn_processor(attn_procs)
    pipe.transformer.config.encoder_hid_dim_type = "ip_image_proj"
    
    # Load and process the input RGB image
    print(f"Processing input RGB image: {args.rgb_image_path}")
    rgb_image = Image.open(args.rgb_image_path).convert("RGB")
    from torchvision import transforms
    to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    rgb_image = to_tensor(rgb_image).unsqueeze(0)
    with torch.no_grad():
        # rgb_embedding = pipe.image_encoder(**rgb_inputs).image_embeds
        # rgb_embedding = rgb_embedding.unsqueeze(0)  # Add batch dimension
        rgb_embedding = image_encoder(rgb_image.to(device, dtype=dtype)).image_embeds
        rgb_embedding = rgb_embedding.unsqueeze(0)
    
    # Load or generate feature map
    if args.feature_map_path and os.path.exists(args.feature_map_path):
        print(f"Loading feature map from: {args.feature_map_path}")
        try:
            feature_map = np.load(args.feature_map_path)
            feature_map = torch.from_numpy(feature_map).float().to(device, dtype=dtype)
            
            feature_map = torch.cat([feature_map[args.refenrence_id:args.refenrence_id+1], feature_map[args.tgt_id:args.tgt_id+1]], dim=1)
            # Ensure correct shape
            # if feature_map.ndim == 3:
            #     if feature_map.shape[0] != feature_channels:
            #         feature_map = feature_map.permute(2, 0, 1)  # Move channels to first dimension if needed
                
            #     # Add batch dimension
            #     feature_map = feature_map.unsqueeze(0)
            # else:
            #     print(f"Warning: Feature map has unexpected dimensions: {feature_map.shape}")
            #     feature_map = torch.randn(1, feature_channels, args.feature_size, args.feature_size, device=device, dtype=dtype)
            
            # Resize if needed
            if feature_map.shape[2] != args.feature_size or feature_map.shape[3] != args.feature_size:
                feature_map = torch.nn.functional.interpolate(
                    feature_map, 
                    size=(args.feature_size, args.feature_size), 
                    mode='bilinear', 
                    align_corners=False
                )
        except Exception as e:
            print(f"Error loading feature map: {e}")
            print("Generating random feature map instead")
            feature_map = torch.randn(1, feature_channels, args.feature_size, args.feature_size, device=device, dtype=dtype)
    else:
        print("Generating random feature map")
        feature_map = torch.randn(1, feature_channels, args.feature_size, args.feature_size, device=device, dtype=dtype)
    
    # Process feature map through encoder
    with torch.no_grad():
        feature_embedding = feature_encoder(feature_map)
        feature_embedding = feature_embedding.unsqueeze(1)  # Add sequence dimension
    
    # Set adapter scales
    for processor in pipe.transformer.attn_processors.values():
        if isinstance(processor, FluxIPAdapterJointAttnProcessor2_0):
            # processor.scale = [rgb_adapter_scale, feature_adapter_scale]
            processor.scale = [0.0, 0.0]
            # processor.adapter_scales = [rgb_adapter_scale, feature_adapter_scale]
    
    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # Run inference
    print(f"Generating {args.num_images} images...")
    for i in range(args.num_images):
        if args.seed is not None and args.num_images > 1:
            # Use a different seed for each image but still reproducible
            generator = torch.Generator(device=device).manual_seed(args.seed + i)
        
        # Generate the image
        output = pipe(
            prompt=args.prompt,
            # negative_prompt=args.negative_prompt,
            joint_attention_kwargs={
                "ip_adapter_image_embeds": [rgb_embedding, feature_embedding],
                # "scale": [0.0, 0.0],
                # "scale": [rgb_adapter_scale, feature_adapter_scale],
                # "adapter_scales": [rgb_adapter_scale, feature_adapter_scale]
            },
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        
        # Save the image
        output_path = os.path.join(args.output_dir, f"output_{i:02d}.png")
        output.images[0].save(output_path)
        print(f"Image saved to {output_path}")
    
    print(f"Generation complete! {args.num_images} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
