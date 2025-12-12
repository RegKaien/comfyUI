#@title Utils Code
# %cd /content/ComfyUI

import os, random, time

import torch
import numpy as np
from PIL import Image
import re, uuid
from nodes import NODE_CLASS_MAPPINGS

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

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

@torch.inference_mode()
def generate(input):
    values = input["input"]
    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt'] # [cite: 1, 2]
    seed = values['seed'] # 0 [cite: 2]
    steps = values['steps'] # 9 [cite: 2]
    cfg = values['cfg'] # 1.0 [cite: 2]
    sampler_name = values['sampler_name'] # euler [cite: 2]
    scheduler = values['scheduler'] # simple [cite: 2]
    denoise = values['denoise'] # 1.0 [cite: 2]
    width = values['width'] # 1024 [cite: 2]
    height = values['height'] # 1024 [cite: 2]
    batch_size = values['batch_size'] # 1.0 [cite: 2]

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)

    positive = CLIPTextEncode.encode(clip, positive_prompt)[0] # [cite: 3]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0] # [cite: 3]
    latent_image = EmptyLatentImage.generate(width, height, batch_size=batch_size)[0] # [cite: 3]
    samples = KSampler.sample(unet, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0] # [cite: 3]
    decoded = VAEDecode.decode(vae, samples)[0].detach() # [cite: 3]
    save_path=get_save_path(positive_prompt)
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save(save_path)
    return save_path,seed
    
import gradio as gr
def generate_ui(
    positive_prompt,
    negative_prompt,
    aspect_ratio,
    seed,
    steps,
    cfg,
    denoise,
    batch_size, # <--- 接收來自 Gradio 介面的 batch_size
    sampler_name="euler", # [cite: 4]
    scheduler="simple" # [cite: 4]
):
    width, height = [int(x) for x in aspect_ratio.split("(")[0].strip().split("x")]

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": int(batch_size), # [cite: 5]
            "seed": int(seed), # [cite: 5]
            "steps": int(steps), # [cite: 5]
            "cfg": float(cfg), # [cite: 5]
            "sampler_name": sampler_name, # [cite: 5]
            "scheduler": scheduler, # [cite: 5]
            "denoise": float(denoise), # [cite: 5]
        }
    }

    image_path,seed = generate(input_data)
    return image_path,image_path,seed


DEFAULT_POSITIVE = """A beautiful woman with platinum blond hair that is almost white, snowy white skin, red bush, very big plump red lips, high cheek bones and sharp. [cite: 6]
She has almond shaped red eyes and she's holding a intricate mask. [cite: 7]
She's wearing white and gold royal gown with a black cloak. [cite: 8]
In the veins of her neck its gold.""" [cite: 9]

DEFAULT_NEGATIVE = """low quality, blurry, unnatural skin tone, bad lighting, pixelated,
noise, oversharpen, soft focus,pixelated""" [cite: 9]

ASPECTS = [
    "1024x1024 (1:1)", "1152x896 (9:7)", "896x1152 (7:9)",
    "1152x864 (4:3)", "864x1152 (3:4)", "1248x832 (3:2)",
    "832x1248 (2:3)", "1280x720 (16:9)", "720x1280 (9:16)",
    "1344x576 (21:9)", "576x1344 (9:21)"
]


custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }" # [cite: 10]

with gr.Blocks(theme=gr.themes.Soft(),css=custom_css) as demo:
  gr.HTML("""
<div style=\"width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;\">
    <h1 style=\"font-size:2.5em; margin-bottom:10px;\">Z-Image-Turbo</h1>
    <a href=\"https://github.com/Tongyi-MAI/Z-Image\" target=\"_blank\">
        <img src=\"https://img.shields.io/badge/GitHub-Z--Image-181717?logo=github&logoColor=white\"
             style=\"height:15px;\">
    </a>
</div>
""")


  with gr.Row():
    with gr.Column():
      positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=5)

      with gr.Row():
        aspect = gr.Dropdown(ASPECTS, value="1024x1024 (1:1)", label="Aspect Ratio")
        seed = gr.Number(value=0, label="Seed (0 = random)", precision=0) # [cite: 11]
        steps = gr.Slider(4, 25, value=9, step=1, label="Steps") # [cite: 11]
      with gr.Row():
        run = gr.Button('🚀 Generate', variant='primary')
      with gr.Accordion('Image Settings', open=False):
        with gr.Row():
          # *** 新增 Batch Size 介面組件 ***
          batch_size_ui = gr.Slider(1, 4, value=1, step=1, label="Batch Size (Images at once)") # <--- 新增
          cfg = gr.Slider(0.5, 4.0, value=1.0, step=0.1, label="CFG") # [cite: 12]
          
        with gr.Row():
          denoise = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Denoise")
        with gr.Row():
          negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3) # [cite: 12]
    with gr.Column():
        download_image=gr.File(label="Download Image")
        output_img = gr.Image(label="Generated Image", height=480)
        used_seed = gr.Textbox(label="Seed Used", interactive=False,show_copy_button=True)

    # *** 更新 run.click 傳入參數 ***
    run.click(
        fn=generate_ui,
        inputs=[positive, negative, aspect, seed, steps, cfg, denoise, batch_size_ui], # <--- 傳遞 batch_size_ui
        outputs=[download_image,output_img, used_seed]
    )

demo.launch(share=True, debug=True)
