#@title Utils Code
# %cd /content/ComfyUI

import os, random, time, sys
import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS
import gradio as gr
import traceback

# --- ComfyUI 核心節點加載 ---
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# --- [修改] Upscale 節點與模型掃描邏輯 ---
upscale_available = False
UpscaleLoaderNode = None
UltimateSDUpscaleNode = None

try:
    # 1. 嘗試加載 Upscale 模型加載器 (通常在 comfy_extras)
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader
    UpscaleLoaderNode = UpscaleModelLoader()
    
    # 2. 嘗試從 NODE_CLASS_MAPPINGS 加載 UltimateSDUpscale
    if "UltimateSDUpscale" in NODE_CLASS_MAPPINGS:
        UltimateSDUpscaleNode = NODE_CLASS_MAPPINGS["UltimateSDUpscale"]()
        upscale_available = True
        print("UltimateSDUpscale node loaded successfully.")
    else:
        print("Warning: 'UltimateSDUpscale' not found in NODE_CLASS_MAPPINGS. Please install ComfyUI_UltimateSDUpscale.")

except ImportError as e:
    print(f"Warning: Could not import necessary nodes. Upscaling will be disabled. Error: {e}")

def get_available_upscalers():
    """掃描 models/upscale_models 資料夾下的模型檔案"""
    path = os.path.join("models", "upscale_models")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return ["None"]
    
    # 支援的副檔名
    valid_extensions = {'.pth', '.pt', '.safetensors', '.bin'}
    files = [f for f in os.listdir(path) if os.path.splitext(f)[1] in valid_extensions]
    
    # 將 'None' 放在第一個選項
    return ["None"] + sorted(files)

# 取得目前可用的模型列表
upscaler_list = get_available_upscalers()
print(f"Found upscalers: {upscaler_list}")

# --- 基礎模型預加載 ---
print("Loading base models...")
# 請確保這些模型檔案存在於您的 models 對應目錄中
with torch.inference_mode():
    try:
        unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
        clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
        vae = VAELoader.load_vae("ae.safetensors")[0]
    except Exception as e:
        print(f"Error loading base models: {e}")
        print("Please ensure checkpoint files exist.")

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
    upscale_model_name = values.get('upscale_model_name', "None") # 接收模型名稱

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    print(f"Encoding prompts...")
    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    
    # 1. 生成 Latent
    print(f"Generating empty latent {width}x{height}...")
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    
    # 2. 採樣 (KSampler)
    print(f"Sampling with {sampler_name}/{scheduler}...")
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    
    # 3. VAE 解碼
    print("Decoding VAE...")
    decoded = VAEDecode.decode(vae, samples)[0]
    
    # --- 4. 執行 Upscale (使用 UltimateSDUpscale) ---
    if upscale_model_name != "None" and upscale_available and UltimateSDUpscaleNode:
        print(f"Upscaling with UltimateSDUpscale using model: {upscale_model_name}...")
        try:
            # 動態加載選定的 Upscale 模型
            current_upscale_model = UpscaleLoaderNode.load_model(upscale_model_name)[0]
            
            # 設定 UltimateSDUpscale 參數
            # 這裡使用固定參數以簡化 UI，您也可以將其改為從 UI 傳入
            usdu_upscale_by = 2.0       # 放大倍率
            usdu_denoise = 0.35         # 重繪幅度 (重要：不要設為 1.0)
            
            # 調用 UltimateSDUpscale
            # 注意：這裡需要傳入原始的 model, positive, negative, vae 供重繪使用
            decoded = UltimateSDUpscaleNode.upscale(
                image=decoded,
                model=unet,
                positive=positive,
                negative=negative,
                vae=vae,
                upscale_by=usdu_upscale_by,
                seed=seed,
                steps=steps,            # 重繪步數
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=usdu_denoise,   # 重繪降噪強度
                upscale_model=current_upscale_model,
                mode_type="Linear",     # 模式: Linear, Chess, None
                tile_width=512,
                tile_height=512,
                mask_blur=8,
                tile_padding=32,
                seam_fix_mode="None",
                seam_fix_denoise=1.0,
                seam_fix_mask_blur=8,
                seam_fix_width=64,
                seam_fix_padding=16,
                force_uniform_tiles=True,
                tiled_decode=False
            )[0]
            
            print("Ultimate Upscale finished.")
        except Exception as e:
            print(f"Error during upscaling: {e}")
            traceback.print_exc()
            print("Returning original size image.")

    # 轉為可保存的格式
    decoded = decoded.detach()
    
    saved_paths = []
    # ComfyUI 的圖像格式是 [Batch, H, W, C] 且數值為 0-1 的 Float Tensor
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
    upscale_model_name, # 接收下拉選單的值
    sampler_name="euler",
    scheduler="simple"
):
    # 解析長寬比字串，例如 "864x1152 (3:4)"
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
            "upscale_model_name": upscale_model_name, # 傳遞模型名稱
        }
    }

    image_paths, seed = generate(input_data)
    return image_paths, image_paths, seed

# --- Gradio 介面定義 ---
DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red blush, very big plump red lips, high cheek bones and sharp features. She has almond shaped red eyes and she's holding a intricate mask.
She's wearing white and gold royal gown with a black cloak.
In the veins of her neck its gold."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus, pixelated"""

ASPECTS = [
    "864x1152 (3:4)", "720x1280 (9:16)", "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", 
    "1344x576 (21:9)", "576x1344 (9:21)"
]

custom_css = """
.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    # Z-Image-Turbo (with Ultimate SD Upscale)
    """)

    with gr.Row():
        # 左側控制欄
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

            # 第一列：尺寸、種子、步數
            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="864x1152 (3:4)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
            
            # 第二列：Batch Size 與 Upscale Model
            with gr.Row():
                batch_size_input = gr.Slider(1, 6, value=1, step=1, label="Batch Size")
                upscale_dropdown = gr.Dropdown(
                    choices=upscaler_list,
                    value="None",
                    label="Upscale Model",
                    info="Files in models/upscale_models/. Using USDU (2x upscale, 0.35 denoise)."
                )

            # Generate 按鈕
            with gr.Row():
                run = gr.Button('Generate', variant='primary')
            
            # 進階設定
            with gr.Accordion('Image Settings', open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Initial Denoise")
                
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
        inputs=[positive, negative, aspect, seed, steps, cfg, denoise, batch_size_input, upscale_dropdown], 
        outputs=[download_image, output_img, used_seed]
    )

demo.launch(share=True, debug=True)
