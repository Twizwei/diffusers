# import torch
# from diffusers import FluxPipeline

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# # prompt = "A cat holding a sign that says hello world"
# # prompt = "A photo of Sagrada Família in Barcelona"
# prompt = "A photo of Brandenburg Gate in Berlin"
# image = pipe(
#     prompt,
#     height=1024,
#     width=1024,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0),
#     # generator=torch.Generator(device="cuda").manual_seed(0)
# ).images[0]
# image.save("flux-dev.png")

# import torch

# import torch
# from huggingface_hub import hf_hub_download, upload_file
# from diffusers import AutoPipelineForText2Image, FluxPipeline
# from safetensors.torch import load_file

# username = "twizwei"
# repo_id = f"{username}/garden-view-Flux-LoRA"

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')


# pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")

# text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
# tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

# # embedding_path = hf_hub_download(repo_id=repo_id, filename="garden-view-Flux-LoRA-Flux-LoRA_emb.safetensors", repo_type="model")
# embedding_path = "/fs/vulcan-projects/sc4d/flux_ckpts/garden-view-Flux-LoRA/garden-view-Flux-LoRA_emb.safetensors"    

# state_dict = load_file(embedding_path)
# # load embeddings of text_encoder 1 (CLIP ViT-L/14)
# pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# # load embeddings of text_encoder 2 (T5 XXL) - ignore this line if you didn't enable `--enable_t5_ti`
# # pipe.load_textual_inversion(state_dict["t5"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

# instance_token = "<s0><s1>"
# # prompt = f"a {instance_token} icon of an orange llama eating ramen, in the style of {instance_token}"
# # prompt = f"a {instance_token} garden view with a bear in it."
# prompt = f"photo of a garden view with {instance_token}"

# image = pipe(prompt=prompt, num_inference_steps=50, joint_attention_kwargs={"scale": 1.0}, ).images[0]
# # image = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=7.0).images[0]
# image.save("garden.png")


import torch
from diffusers import FluxPipeline, AutoPipelineForText2Image
from diffusers.utils import load_image
from safetensors.torch import load_file

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
).to("cuda")

# username = "twizwei"
# repo_id = f"{username}/garden-view-Flux-LoRA"
# pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")
# embedding_path = "/fs/vulcan-projects/sc4d/flux_ckpts/garden-view-Flux-LoRA/garden-view-Flux-LoRA_emb.safetensors"  
# state_dict = load_file(embedding_path)
# pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
# instance_token = "<s0><s1>"

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flux_ip_adapter_input.jpg").resize((1024, 1024))

pipe.load_ip_adapter(
    "XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
)
pipe.set_ip_adapter_scale(1.0)

image = pipe(
    width=1024,
    height=1024,
    # prompt="wearing sunglasses",
    prompt="sitting in a bench in Florence",
    # prompt=f"standing in a {instance_token}",
    negative_prompt="",
    true_cfg_scale=4.0,
    generator=torch.Generator().manual_seed(4444),
    ip_adapter_image=image,
).images[0]

image.save('./inference_results/flux_ip_adapter_output.jpg')