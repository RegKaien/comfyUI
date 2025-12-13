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

# --- ComfyUI 節點加載 ---
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# --- [新增] Upscale 相關節點 ---
UpscaleModelLoader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
ImageUpscaleWithModel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()

# --- 模型預加載 ---
print("Loading models...")
with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]
    
    # --- [新增] 加載 4x Upscale 模型 ---
    # 請確保 '4x-UltraSharp.pth' 存在於 models/upscale_models/ 目錄下
    # 如果使用不同的 Digital Art 模型 (如 4x_NMKD-Siax_200k.pth)，請修改此處檔名
    try:
        upscale_net = UpscaleModelLoader.load_model("4x-UltraSharp.pth")[0]
        print("Upscale model loaded successfully.")
    except Exception as e:
        print(f"Warning: Upscale model not found or failed to load. Upscaling may fail if enabled. Error: {e}")
        upscale_net = None

save_dir="./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt):
  save_dir = "./results"
  # 簡單過濾檔名非法字元
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
    use_upscale = values['use_upscale'] # [新增] 接收 upscale 參數

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    
    # 生成 Latent
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    
    # 採樣 (Sampling)
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    
    # VAE 解碼 (Latent -> Pixel)
    decoded = VAEDecode.decode(vae, samples)[0]
    
    # --- [新增] 執行 Upscale 邏輯 ---
    if use_upscale and upscale_net is not None:
        print("Upscaling images...")
        # ImageUpscaleWithModel 接收 (upscale_model, image)
        decoded = ImageUpscaleWithModel.upscale(upscale_net, decoded)[0]
    elif use_upscale and upscale_net is None:
        print("Upscale requested but model not loaded. Skipping.")

    # 轉為可保存的格式
    decoded = decoded.detach()
    
    # 保存圖片
    saved_paths = []
    # 將 Tensor 轉為 Numpy 陣列: (Batch_Size, Height, Width, Channels)
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
    use_upscale, # [新增] 接收 UI 參數
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
            "use_upscale": use_upscale, # [新增] 傳遞參數
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
        # 左側控制欄
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
                
                # [新增] Upscale Checkbox
                with gr.Row():
                    use_upscale = gr.Checkbox(label="Enable 4x Upscale (Digital Art Sharp)", value=False, info="Increases generation time significantly.")
                
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)
        
        # 右側顯示欄
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

    # 事件綁定
    run.click(
        fn=generate_ui,
        # [新增] 將 use_upscale 放入 inputs
        inputs=[positive, negative, aspect, seed, steps, cfg, denoise, batch_size_input, use_upscale], 
        outputs=[download_image, output_img, used_seed]
    )

demo.launch(share=True, debug=True)
