#@title Utils Code
# %cd /content/ComfyUI

import os, random, time, sys
import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS
import gradio as gr

# --- ComfyUI 核心節點加載 ---
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# --- Upscale 節點與模型掃描邏輯 ---
upscale_available = False
UpscaleLoaderNode = None
ImageUpscaleNode = None

try:
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
    UpscaleLoaderNode = UpscaleModelLoader()
    ImageUpscaleNode = ImageUpscaleWithModel()
    upscale_available = True
    print("Upscale nodes imported successfully.")
except ImportError:
    print("Warning: Could not import Upscale nodes from comfy_extras. Upscaling will be disabled.")

def get_available_upscalers():
    """掃描 models/upscale_models 資料夾下的模型檔案"""
    path = os.path.join("models", "upscale_models")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return ["None"]
    
    valid_extensions = {'.pth', '.pt', '.safetensors', '.bin'}
    files = [f for f in os.listdir(path) if os.path.splitext(f)[1] in valid_extensions]
    return ["None"] + sorted(files)

upscaler_list = get_available_upscalers()
print(f"Found upscalers: {upscaler_list}")

# --- 基礎模型預加載 (修改處：更換為 Ideogram 4 模型) ---
print("Loading Ideogram 4 base models...")
with torch.inference_mode():
    # 根據 Ideogram 4 實際權重格式調整，這裡預設使用標準 fp8 格式
    unet = UNETLoader.load_unet("ideogram-4-fp8.safetensors", "fp8_e4m3fn")[0]
    # 調整為 Ideogram 4 的文本編碼器類型 (此處 type 依據 ComfyUI 支援結構調整，暫定為 default)
    clip = CLIPLoader.load_clip("ideogram-4-text-encoder.safetensors", type="default")[0]
    vae = VAELoader.load_vae("ideogram-4-vae.safetensors")[0]

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
    start_time = time.time()
    upscale_duration = 0.0

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
    upscale_model_name = values.get('upscale_model_name', "None")

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    decoded = VAEDecode.decode(vae, samples)[0]
    
    if upscale_model_name != "None" and upscale_available:
        print(f"Upscaling with model: {upscale_model_name}...")
        upscale_start = time.time()
        try:
            current_upscale_model = UpscaleLoaderNode.load_model(upscale_model_name)[0]
            decoded = ImageUpscaleNode.upscale(current_upscale_model, decoded)[0]
            print("Upscale finished.")
        except Exception as e:
            print(f"Error during upscaling: {e}")
            print("Returning original size image.")
        upscale_end = time.time()
        upscale_duration = upscale_end - upscale_start

    decoded = decoded.detach()
    
    saved_paths = []
    images_np = np.array(decoded * 255, dtype=np.uint8)
    
    for img_np in images_np:
        save_path = get_save_path(positive_prompt)
        Image.fromarray(img_np).save(save_path)
        saved_paths.append(save_path)

    total_end_time = time.time()
    total_duration = total_end_time - start_time
        
    return saved_paths, seed, total_duration, upscale_duration

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
    upscale_model_name,
    sampler_name="euler",
    scheduler="normal" # 修改處：由 simple 改為非 Turbo 經常用的 normal 
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
            "upscale_model_name": upscale_model_name,
        }
    }

    image_paths, seed, total_duration, upscale_duration = generate(input_data)
    
    if upscale_model_name != "None":
        time_info = f"⏱️ Total: {total_duration:.2f}s | 🔍 Upscale: {upscale_duration:.2f}s"
    else:
        time_info = f"⏱️ Total: {total_duration:.2f}s | (Upscale skipped)"

    return image_paths, image_paths, seed, time_info

# --- Gradio 介面定義 ---
DEFAULT_POSITIVE = """masterpiece, best quality, amazing quality, absurdres, (realistic), beautiful and aesthetic, looking at viewer, 1girl,
A beautiful woman with dark hair, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp. 
She's wearing white and gold royal gown with a black cloak."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated, ((face out of frame)), ((more than two arms))"""

ASPECTS = [
    "1024x1024 (1:1)", "864x1152 (3:4)", "720x1280 (9:16)", "1152x864 (4:3)", "1280x720 (16:9)"
]

custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks() as demo:
    # 修改處：更新 UI 標題
    gr.HTML("""
    # Ideogram 4 (with Upscaler)
    """)

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

            with gr.Row():
                run = gr.Button('Generate', variant='primary')
            
            download_image = gr.File(label="Download Image(s)")
                
            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                # 修改處：將預設步數調高至 25（原先 Turbo 僅 9）
                steps = gr.Slider(10, 50, value=25, step=1, label="Steps")
            
            with gr.Row():
                batch_size_input = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
                upscale_dropdown = gr.Dropdown(
                    choices=upscaler_list,
                    value="NMKD_2x_CX_100k.pth",
                    label="Upscale Model",
                    info="Files detected in models/upscale_models/"
                )
           
            with gr.Accordion('Image Settings', open=False):
                with gr.Row():
                    # 修改處：將預設 CFG 調高至 5.0（原先 Turbo 僅 1.0）
                    cfg = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)
        
        with gr.Column():
            output_img = gr.Gallery(
                label="Generated Images", 
                show_label=True, 
                elem_id="gallery", 
                columns=2, 
                rows=2, 
                height=600,
                object_fit="contain"
            )

            used_seed = gr.Textbox(label="Seed Used", interactive=False)
            performance_info = gr.Textbox(label="Performance Stats", interactive=False)

    run.click(
        fn=generate_ui,
        inputs=[positive, negative, aspect, seed, steps, cfg, denoise, batch_size_input, upscale_dropdown], 
        outputs=[download_image, output_img, used_seed, performance_info]
    )

demo.launch(share=True, debug=True, theme=gr.themes.Soft(), css=custom_css)
