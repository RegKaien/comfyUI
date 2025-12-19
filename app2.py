#@title Utils Code
# %cd /content/ComfyUI

import os, random, time, sys, importlib

import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS
import nodes

# ==========================================
# 1. 載入標準節點 (Standard Nodes)
# ==========================================
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

# ==========================================
# 2. 載入 UpscaleModelLoader (來自 comfy_extras)
# ==========================================
try:
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader
    UpscaleModelLoader = UpscaleModelLoader()
except ImportError:
    print("⚠️ Warning: Could not import UpscaleModelLoader from comfy_extras.")
    # 嘗試從 mappings 找 (fallback)
    if "UpscaleModelLoader" in NODE_CLASS_MAPPINGS:
        UpscaleModelLoader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
    else:
        UpscaleModelLoader = None

# ==========================================
# 3. 載入 UltimateSDUpscale (Custom Node)
# ==========================================
UltimateSDUpscale = None

def load_custom_node_by_name(target_node_name, folder_hint="UltimateSDUpscale"):
    """嘗試從 custom_nodes 資料夾載入指定的插件"""
    custom_nodes_path = os.path.join(os.getcwd(), "custom_nodes")
    if not os.path.exists(custom_nodes_path):
        return None

    # 1. 先檢查 NODE_CLASS_MAPPINGS 是否已經有了
    if target_node_name in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS[target_node_name]

    # 2. 搜尋符合 hint 的資料夾並嘗試執行 __init__.py
    for item in os.listdir(custom_nodes_path):
        if folder_hint.lower() in item.lower() and os.path.isdir(os.path.join(custom_nodes_path, item)):
            node_path = os.path.join(custom_nodes_path, item)
            init_file = os.path.join(node_path, "__init__.py")
            if os.path.exists(init_file):
                try:
                    # 動態導入模組
                    spec = importlib.util.spec_from_file_location(item, init_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[item] = module
                    spec.loader.exec_module(module)
                    
                    # 更新 MAPPINGS
                    if hasattr(module, "NODE_CLASS_MAPPINGS"):
                        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                        if target_node_name in NODE_CLASS_MAPPINGS:
                            print(f"✅ Successfully loaded custom node: {item}")
                            return NODE_CLASS_MAPPINGS[target_node_name]
                except Exception as e:
                    print(f"❌ Error loading custom node {item}: {e}")
    return None

# 嘗試載入 UltimateSDUpscale
USDU_Class = load_custom_node_by_name("UltimateSDUpscale", folder_hint="Ultimate")
if USDU_Class:
    UltimateSDUpscale = USDU_Class()
else:
    print("⚠️ Warning: UltimateSDUpscale node NOT found. Upscaling functions will be disabled.")


# ==========================================
# 4. 模型與參數設定
# ==========================================

# 預設 Upscale 模型列表 (請確保 models/upscale_models 路徑下有這些檔案)
UPSCALE_MODELS = [
    "None", 
    "4x-UltraSharp.pth", 
    "RealESRGAN_x4plus.pth", 
    "RealESRGAN_x4plus_anime_6B.pth",
    "R-ESRGAN 4x+ Anime6B.pth"
]

USDU_MODES = ["Linear", "Chess", "None"]
SEAM_FIX_MODES = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]

with torch.inference_mode():
    try:
        unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
        clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
        vae = VAELoader.load_vae("ae.safetensors")[0]
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure checkpoints are in 'models/checkpoints', 'models/clip', 'models/vae'")

save_dir="./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt, suffix=""):
  save_dir = "./results"
  safe_prompt = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
  uid = uuid.uuid4().hex[:6]
  filename = f"{safe_prompt}_{uid}{suffix}.png"
  path = os.path.join(save_dir, filename)
  return path

@torch.inference_mode()
def generate(input_data):
    values = input_data["input"]
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

    # Upscale 參數
    upscale_model_name = values.get('upscale_model_name', "None")
    upscale_by = values.get('upscale_by', 1.0)
    usdu_denoise = values.get('usdu_denoise', 0.2)
    usdu_steps = values.get('usdu_steps', 20)
    usdu_cfg = values.get('usdu_cfg', 8.0)
    # 這裡預設使用與 base generation 相同的 sampler/scheduler，或者 UI 傳入
    usdu_sampler = values.get('sampler_name', "euler") 
    usdu_scheduler = values.get('scheduler', "normal")
    
    mode_type = values.get('mode_type', "Linear")
    tile_width = values.get('tile_width', 512)
    tile_height = values.get('tile_height', 512)
    mask_blur = values.get('mask_blur', 8)
    tile_padding = values.get('tile_padding', 32)
    seam_fix_mode = values.get('seam_fix_mode', "None")
    seam_fix_denoise = values.get('seam_fix_denoise', 1.0)
    seam_fix_width = values.get('seam_fix_width', 64)
    seam_fix_mask_blur = values.get('seam_fix_mask_blur', 8)
    seam_fix_padding = values.get('seam_fix_padding', 16)
    
    # 修正部分 boolean 參數，Gradio 可能傳回字串
    force_uniform_tiles = True
    tiled_decode = False

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    print(f"Generating image with seed: {seed}")

    # 1. 基礎生成 (Base Generation)
    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0]
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
    decoded = VAEDecode.decode(vae, samples)[0].detach() # Tensor [B, H, W, C]

    final_image = decoded

    # 2. Ultimate SD Upscale 流程
    # 只有當 UpscaleModelLoader 和 UltimateSDUpscale 都成功載入，且用戶選擇了模型時才執行
    if upscale_model_name != "None" and upscale_by > 1.0:
        if UpscaleModelLoader is None or UltimateSDUpscale is None:
            print("❌ Skipping Upscale: UpscaleModelLoader or UltimateSDUpscale node is missing.")
        else:
            print(f"🔄 Starting Ultimate SD Upscale ({upscale_by}x) with model: {upscale_model_name}...")
            try:
                # 載入 Upscale Model
                # UpscaleModelLoader.load_model 返回 (model, )
                upscale_model = UpscaleModelLoader.load_model(upscale_model_name)[0]
                
                # 執行 Upscale
                upscaled_result = UltimateSDUpscale.upscale(
                    image=decoded,
                    model=unet,
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    upscale_by=upscale_by,
                    seed=seed,
                    steps=usdu_steps,
                    cfg=usdu_cfg,
                    sampler_name=usdu_sampler,
                    scheduler=usdu_scheduler,
                    denoise=usdu_denoise,
                    upscale_model=upscale_model,
                    mode_type=mode_type,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    mask_blur=mask_blur,
                    tile_padding=tile_padding,
                    seam_fix_mode=seam_fix_mode,
                    seam_fix_denoise=seam_fix_denoise,
                    seam_fix_mask_blur=seam_fix_mask_blur,
                    seam_fix_width=seam_fix_width,
                    seam_fix_padding=seam_fix_padding,
                    force_uniform_tiles=force_uniform_tiles,
                    tiled_decode=tiled_decode
                )[0]
                final_image = upscaled_result
                print("✅ Upscale finished.")
            except Exception as e:
                print(f"❌ Upscale failed: {e}")
                import traceback
                traceback.print_exc()

    save_path = get_save_path(positive_prompt)
    Image.fromarray(np.array(final_image * 255, dtype=np.uint8)[0]).save(save_path)
    return save_path, seed

# ==========================================
# 5. Gradio UI
# ==========================================
import gradio as gr

def generate_ui(
    positive_prompt,
    negative_prompt,
    aspect_ratio,
    seed,
    steps,
    cfg,
    denoise,
    # Upscale UI 參數
    upscale_model_name,
    upscale_by,
    usdu_denoise,
    usdu_steps,
    usdu_cfg,
    mode_type,
    tile_width,
    tile_height,
    seam_fix_mode,
    seam_fix_denoise,
    seam_fix_width,
    batch_size=1,
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
            
            # 傳遞 Upscale 參數
            "upscale_model_name": upscale_model_name,
            "upscale_by": float(upscale_by),
            "usdu_denoise": float(usdu_denoise),
            "usdu_steps": int(usdu_steps),
            "usdu_cfg": float(usdu_cfg),
            "mode_type": mode_type,
            "tile_width": int(tile_width),
            "tile_height": int(tile_height),
            "seam_fix_mode": seam_fix_mode,
            "seam_fix_denoise": float(seam_fix_denoise),
            "seam_fix_width": int(seam_fix_width),
        }
    }

    image_path, used_seed = generate(input_data)
    return image_path, image_path, used_seed

DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp.
She has almond shaped red eyes and she's holding a intricate mask.
She's wearing white and gold royal gown with a black cloak.
In the veins of her neck its gold."""

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus,pixelated"""

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]

custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    <div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
        <h1 style="font-size:2.5em; margin-bottom:10px;">Z-Image-Turbo + Ultimate SD Upscale</h1>
        <a href="https://github.com/Tongyi-MAI/Z-Image" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white" style="height:15px;">
        </a>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)
            
            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
            
            with gr.Row():
                run = gr.Button('🚀 Generate', variant='primary')
            
            with gr.Accordion('Image Settings', open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)
            
            # 新增 Ultimate SD Upscale 設定區塊
            with gr.Accordion('Ultimate SD Upscale Settings', open=True):
                with gr.Row():
                    upscale_model_name = gr.Dropdown(choices=UPSCALE_MODELS, value="None", label="Upscale Model")
                    upscale_by = gr.Slider(1.0, 4.0, value=1.0, step=0.5, label="Upscale By")
                
                with gr.Group():
                    with gr.Row():
                        usdu_denoise = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="USDU Denoise")
                        usdu_steps = gr.Slider(10, 100, value=20, step=1, label="USDU Steps")
                        usdu_cfg = gr.Slider(1.0, 20.0, value=8.0, step=0.5, label="USDU CFG")
                    
                    with gr.Row():
                        mode_type = gr.Dropdown(choices=USDU_MODES, value="Linear", label="Mode Type")
                        tile_width = gr.Number(value=512, label="Tile Width", precision=0)
                        tile_height = gr.Number(value=512, label="Tile Height", precision=0)

                    with gr.Row():
                        seam_fix_mode = gr.Dropdown(choices=SEAM_FIX_MODES, value="None", label="Seam Fix Mode")
                        seam_fix_denoise = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Seam Fix Denoise")
                        seam_fix_width = gr.Number(value=64, label="Seam Fix Width", precision=0)

        with gr.Column():
            download_image = gr.File(label="Download Image")
            output_img = gr.Image(label="Generated Image", height=480)
            used_seed = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)

    run.click(
        fn=generate_ui,
        inputs=[
            positive, negative, aspect, seed, steps, cfg, denoise,
            upscale_model_name, upscale_by, usdu_denoise, usdu_steps, usdu_cfg,
            mode_type, tile_width, tile_height, seam_fix_mode, seam_fix_denoise, seam_fix_width
        ],
        outputs=[download_image, output_img, used_seed]
    )

demo.launch(share=True, debug=True)
