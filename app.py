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

# --- 基礎模型預加載 ---
print("Loading base models...")
with torch.inference_mode():
    unet = UNETLoader.load_unet("z-image-turbo-fp8-e4m3fn.safetensors", "fp8_e4m3fn_fast")[0]
    clip = CLIPLoader.load_clip("qwen_3_4b.safetensors", type="lumina2")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]

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
She's wearing white and gold royal gown with a black cloak.  In the veins of her neck its gold,
(royal palace), prestige, gorgeous, luxury, jewelry, gem, hyper-detailed, fractal art, gold complex fractal patterns, fibonacci set, PEONY, Lotus, vivid [red & white ] colors, sleek, 
highly detailed, (36F huge breasts:1.3), off shoulders, ((stockings)), zentangle, mandala, tangle, entangle, the most beautiful form of chaos, elegant, a brutalist designed, (vivid  colors), 
romanticism, updo, thigh highs, dynamic pose, intimidating, high cut, deep-V, huge necklace, huge earrings, ornate, huge hair accessory, """

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus, pixelated, (((mutation))), mutated, ((bad anatomy)), (((bad proportions))), (((disfigured))), ((deformed)), ((mutilated)), ((morbid)), ((extra limbs)), 
(malformed limbs), ((poorly drawn hands)), (((distorted hands))), (((extra hands))), ((mutated hands)), (((fused fingers))) """

ASPECTS = [
    "864x1152 (3:4)", "720x1280 (9:16)", "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", 
    "1344x576 (21:9)", "576x1344 (9:21)"
]

custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    # Z-Image-Turbo (with Upscaler)
    """)

    with gr.Row():
        # 左側控制欄
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

            # Generate 按鈕
            with gr.Row():
                run = gr.Button('Generate', variant='primary')
            
            # [修改] 下載元件移到這裡 (Generate 按鈕下方)
            download_image = gr.File(label="Download Image(s)")
                
            # 第一列：尺寸、種子、步數
            with gr.Row():
                aspect = gr.Dropdown(ASPECTS, value="864x1152 (3:4)", label="Aspect Ratio")
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(4, 25, value=9, step=1, label="Steps")
            
            # 第二列：Batch Size 與 Upscale Model
            with gr.Row():
                batch_size_input = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
                upscale_dropdown = gr.Dropdown(
                    choices=upscaler_list,
                    value="NMKD_2x_CX_100k.pth",
                    label="Upscale Model",
                    info="Files detected in models/upscale_models/"
                )
           
            # 進階設定
            with gr.Accordion('Image Settings', open=False):
                with gr.Row():
                    cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0.1, 1.0, value=0.85, step=0.05, label="Denoise")
                
                with gr.Row():
                    negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)
        
        # 右側顯示欄
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

            # [已移除] 原本在這裡的 download_image 
            used_seed = gr.Textbox(label="Seed Used", interactive=False, show_copy_button=True)
            performance_info = gr.Textbox(label="Performance Stats", interactive=False)

    # 事件綁定
    run.click(
        fn=generate_ui,
        inputs=[positive, negative, aspect, seed, steps, cfg, denoise, batch_size_input, upscale_dropdown], 
        outputs=[download_image, output_img, used_seed, performance_info]
    )

demo.launch(share=True, debug=True)
