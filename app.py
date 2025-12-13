#@title Utils Code
# %cd /content/ComfyUI

import os, random, time
import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS
import gradio as gr
import sys

# --- ComfyUI 核心節點加載 ---
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# --- [修正] Upscale 節點加載邏輯 ---
# 這些節點通常在庫存的 comfy_extras 中，不在主 mappings 裡
upscale_available = False
try:
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
    # 實例化節點類別
    UpscaleLoaderNode = UpscaleModelLoader()
    ImageUpscaleNode = ImageUpscaleWithModel()
    upscale_available = True
    print("Upscale nodes imported successfully.")
except ImportError:
    print("Warning: Could not import Upscale nodes from comfy_extras. Upscaling will be disabled.")
    UpscaleLoaderNode = None
    ImageUpscaleNode = None

# --- 模型預加載 ---
print("Loading models...")
with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]
    
    # --- 加載 4x Upscale 模型 ---
    upscale_net = None
    if upscale_available:
        # 設定目標模型名稱 (Digital Art Sharp 常用 4x-UltraSharp)
        # 請確認 models/upscale_models/ 中有此檔案，若無可改為其他可用模型
        target_upscale_model = "4x-UltraSharp.pth" 
        
        # 檢查檔案是否存在，避免報錯
        model_path = os.path.join("models", "upscale_models", target_upscale_model)
        if os.path.exists(model_path):
            try:
                upscale_net = UpscaleLoaderNode.load_model(target_upscale_model)[0]
                print(f"Upscale model '{target_upscale_model}' loaded.")
            except Exception as e:
                print(f"Failed to load upscale model: {e}")
        else:
            print(f"Upscale model file '{target_upscale_model}' not found in models/upscale_models/. Upscaling will be skipped.")

save_dir="./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt):
  save_dir = "./results"
  safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
  uid = uuid.uuid4().hex[:6]
  filename = f"{safe_prompt}_{uid}.png"
  path = os.path.join(save_dir, filename)
  return path

# --- 核心生成邏輯 ---
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
    batch_size = values['batch_size']
    use_upscale = values.get('use_upscale', False)

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    decoded = VAEDecode.decode(vae, samples)[0]
    
    # --- 執行 Upscale 邏輯 ---
    if use_upscale and upscale_net is not None and upscale_available:
        print("Upscaling images...")
        try:
            # 進行放大
            decoded = ImageUpscaleNode.upscale(upscale_net, decoded)[0]
        except Exception as e:
            print(f"Upscale failed during generation: {e}")
    elif use_upscale:
        print("Upscale requested but model not loaded. Skipping.")

    decoded = decoded.detach()
    
    saved_paths = []
    images_np = np.array(decoded * 255, dtype=np.uint8)
    
    for img_np in images_np:
        save_path = get_save_path(positive_prompt)
        Image.fromarray(img_np).save(save_path)
        saved_paths.append(save_path)
        
    return saved_paths, seed

# --- UI 邏輯中介 ---
def generate_ui(
    positive_prompt,
    negative_prompt,
    aspect_ratio,
    seed,
    steps,
    cfg,
    denoise,
    batch_size,
    use_upscale,
    sampler_name="euler",
    scheduler="simple"
):
    width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "use_upscale": use_upscale,
        }
    }

    image_paths, seed = generate(input_data)
    return image_paths, image_paths, seed

# --- Gradio 介面定義 ---
DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp. She has almond shaped red eyes and she's holding a intricate mask. She's wearing white and gold royal gown with a black cloak.  In the veins of her neck its gold."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus,pixelated"""

ASPECTS = [
    "864x1152 (3:4)", "720x1280 (9:16)", "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", 
    "1344x576 (21:9)", "576x1344 (9:21)"
]

custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    # Z-Image-Turbo (with 4x Upscale)
    """)

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="864x1152 (3:4)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
            
            with gr.Row():
                run = gr.Button('Generate', variant='primary')
            
            with gr.Accordion('Image Settings', open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                    batch_size_input = gr.Slider(1, 6, value=2, step=1, label="Batch Size")
                
                with gr.Row():
                    # Upscale 開關
                    use_upscale = gr.Checkbox(
                        label="Enable 4x Upscale (Digital Art Sharp)", 
                        value=False, 
                        info="Requires '4x-UltraSharp.pth' in models/upscale_models/"
                    )
                
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)
        
        with gr.Column():
            download_image = gr.File(label="Download Image(s)")
            
            output_img = gr.Gallery(
                label="Generated Images", 
                show_label=True, 
                elem_id="gallery", 
                columns=2, 
                rows=2, 
                height=600,
                object_fit="contain"
            )
            
            used_seed = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)

    run.click(
        fn=generate_ui,
        inputs=[positive, negative, aspect, seed, steps, cfg, denoise, batch_size_input, use_upscale], 
        outputs=[download_image, output_img, used_seed]
    )

demo.launch(share=True, debug=True)
