#@title Utils Code
# %cd /content/ComfyUI
# https://raw.githubusercontent.com/NeuralFalconYT/Z-Image-Colab/refs/heads/main/app.py
# https://github.com/RegKaien/comfyUI/raw/refs/heads/main/app.py

import os, random, time
import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS
import gradio as gr

# --- ComfyUI 節點初始化 ---
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# --- 模型載入 ---
with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt):
    save_dir = "./results"
    safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
    uid = uuid.uuid4().hex[:6]
    filename = f"{safe_prompt}_{uid}.png"
    path = os.path.join(save_dir, filename)
    return path

# --- 後端生成邏輯 ---
@torch.inference_mode()
def generate(input):
    values = input["input"]
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']
    denoise = values['denoise']
    width = values['width']
    height = values['height']
    batch_size = values['batch_size']  # 取得 batch_size

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    
    # 這裡傳入 batch_size 以生成多張 Latent
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_
