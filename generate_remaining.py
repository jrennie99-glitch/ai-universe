#!/usr/bin/env python3
"""Generate remaining 5 portraits."""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

device = "mps"
OUTPUT_DIR = os.path.expanduser("~/.openclaw/workspace/ai-world-3d/public/portraits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

remaining = [
    {"id": "keisha", "name": "Keisha Campbell", "prompt": "photorealistic portrait of a beautiful young Jamaican woman with deep dark skin, athletic body, long braided hair, stunning expression, wearing an orange crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "aliyah", "name": "Aliyah Davis", "prompt": "photorealistic portrait of a gorgeous young black American woman with dark skin, athletic body, long natural hair, warm smile, wearing an orange fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "jasmine", "name": "Jasmine Okafor", "prompt": "photorealistic portrait of a beautiful young Nigerian woman with dark skin, athletic body, long natural hair, confident expression, wearing a light blue top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "emma", "name": "Emma Wilson", "prompt": "photorealistic portrait of a beautiful young American woman with light skin, fit body, long blonde hair, bright smile, wearing a purple fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "hanna", "name": "Hanna Brooks", "prompt": "photorealistic portrait of a gorgeous young American woman with light skin, fit body, long auburn hair, warm expression, wearing a green crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
]

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

for i, char in enumerate(remaining):
    outpath = os.path.join(OUTPUT_DIR, f"{char['id']}.png")
    if os.path.exists(outpath):
        print(f"[{i+1}/{len(remaining)}] {char['name']} - exists, skip")
        continue
    print(f"[{i+1}/{len(remaining)}] {char['name']}...")
    image = pipe(char['prompt'], num_inference_steps=25, guidance_scale=7.5, width=512, height=768, generator=torch.Generator(device=device).manual_seed(99+i)).images[0]
    image.save(outpath)
    print(f"  ✅ Done")

print("All 30 portraits complete!")
