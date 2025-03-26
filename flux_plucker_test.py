import torch
import torch.nn as nn
from PIL import Image
from diffusers import FluxPipeline
from diffusers.models.embeddings import ImageProjection, MultiIPAdapterImageProjection
from diffusers.models.attention_processor import FluxIPAdapterJointAttnProcessor2_0
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

class CustomFeatureEncoder(nn.Module):
    """Simple encoder for 12-dim feature maps"""
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

def main():
    # 1. Create the base pipeline with image encoder and feature extractor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16
        ),
        feature_extractor=CLIPImageProcessor()
    ).to(device)

    # 2. Create custom encoder for feature maps
    feature_encoder = CustomFeatureEncoder(
        in_channels=12,
        hidden_dim=256,
        out_dim=768  # Match CLIP embedding dimension
    ).to(device).to(torch.float16).eval()

    # 3. Create IP-Adapters (move to device)
    rgb_adapter = ImageProjection(
        cross_attention_dim=4096,
        image_embed_dim=768,
        num_image_text_embeds=4
    ).to(device).to(torch.float16)

    feature_adapter = ImageProjection(
        cross_attention_dim=4096,
        image_embed_dim=768,
        num_image_text_embeds=4
    ).to(device).to(torch.float16)

    # 4. Combine adapters
    multi_adapter = MultiIPAdapterImageProjection([rgb_adapter, feature_adapter]).to(device).to(torch.float16)

    # 5. Set up attention processors
    attn_procs = {}
    for name in pipe.transformer.attn_processors.keys():
        if name.startswith("single_transformer_blocks"):
            attn_processor_class = pipe.transformer.attn_processors[name].__class__
            attn_procs[name] = attn_processor_class()
        else:
            attn_procs[name] = FluxIPAdapterJointAttnProcessor2_0(
                hidden_size=3072,
                cross_attention_dim=4096,
                scale=1.0,
                num_tokens=[4, 4],
                dtype=torch.float16,
                device=device
            )

    # 6. Register components
    pipe.transformer.encoder_hid_proj = multi_adapter
    pipe.transformer.set_attn_processor(attn_procs)
    pipe.transformer.config.encoder_hid_dim_type = "ip_image_proj"

    # 7. Test with sample inputs
    # Load and process RGB image using pipeline's built-in processors
    rgb_image = Image.open("./inference_results/flux-img2img.png").convert("RGB")
    rgb_inputs = pipe.feature_extractor(images=rgb_image, return_tensors="pt").to(device, dtype=torch.float16)
    with torch.no_grad():
        rgb_embedding = pipe.image_encoder(**rgb_inputs).image_embeds.to(device)
        rgb_embedding = rgb_embedding[None, :]

    # Create and process feature map
    feature_map = torch.randn(1, 12, 64, 64, device=device, dtype=torch.float16)
    with torch.no_grad():
        feature_embedding = feature_encoder(feature_map)
        feature_embedding = feature_embedding[None, :]
        
    # 8. Generate image
    output = pipe(
        prompt="a photo of a garden view",
        ip_adapter_image_embeds=[rgb_embedding, feature_embedding],
        joint_attention_kwargs={"scale": 1.0},
        num_inference_steps=20
    )

    # Save output
    output.images[0].save("./inference_results/adapter_test_output.png")
    # print('output image shape: ', output.images[0].size)

    # 9. Try different scales for each adapter
    pipe.set_ip_adapter_scale([0.7, 0.3])  # RGB adapter at 0.7, feature adapter at 0.3
    output = pipe(
        prompt="a photo of a garden view",
        ip_adapter_image_embeds=[rgb_embedding, feature_embedding],
        joint_attention_kwargs={"scale": 1.0},
        num_inference_steps=20
    )
    output.images[0].save("./inference_results/adapter_test_output_weighted.png")
    # print('output image shape: ', output.images[0].size)

    # 10. Try turning off adapters
    pipe.set_ip_adapter_scale([0.0, 0.0])  # RGB adapter at 0.7, feature adapter at 0.3
    output = pipe(
        prompt="a photo of a garden view",
        ip_adapter_image_embeds=[rgb_embedding, feature_embedding],
        joint_attention_kwargs={"scale": 1.0},
        num_inference_steps=20
    )
    output.images[0].save("./inference_results/adapter_test_output_no_adapters.png")

if __name__ == "__main__":
    main()