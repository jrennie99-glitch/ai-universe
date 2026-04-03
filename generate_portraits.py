#!/usr/bin/env python3
"""Generate photorealistic character portraits for AI Universe using Stable Diffusion on MPS."""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
from PIL import Image

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

OUTPUT_DIR = os.path.expanduser("~/.openclaw/workspace/ai-world-3d/public/portraits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Character definitions with detailed prompts
characters = [
    # MEN
    {"id": "omar", "name": "Omar Campbell", "prompt": "photorealistic portrait of a handsome young Jamaican black man with athletic build, short dreadlocks, warm dark skin, confident smile, wearing a cyan fitted t-shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "tony", "name": "Tony Blake", "prompt": "photorealistic portrait of an attractive young Jamaican black man with brown skin, short fade haircut, charismatic expression, wearing an orange fitted shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "marcus", "name": "Marcus Johnson", "prompt": "photorealistic portrait of a handsome black American man with dark skin, broad shoulders, clean shaven head, serious confident expression, wearing a blue dress shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "kwame", "name": "Kwame Asante", "prompt": "photorealistic portrait of a handsome young Ghanaian man with deep dark skin, athletic build, short curly hair, warm smile, wearing an orange polo shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "deon", "name": "Deon Carter", "prompt": "photorealistic portrait of a handsome mixed-race man with brown skin, broad build, neat beard, confident expression, wearing a purple dress shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "carlos", "name": "Carlos Rivera", "prompt": "photorealistic portrait of a handsome young Colombian man with tan skin, athletic build, dark wavy hair, warm smile, wearing a green fitted t-shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "miguel", "name": "Miguel Torres", "prompt": "photorealistic portrait of a handsome young Mexican man with medium skin, lean build, black hair styled neatly, friendly smile, wearing a teal polo shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "jin", "name": "Jin Park", "prompt": "photorealistic portrait of a handsome young Korean man with light skin, lean build, stylish modern haircut, warm expression, wearing a pink fitted shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "rafael", "name": "Rafael Santos", "prompt": "photorealistic portrait of a handsome young Brazilian man with bronze skin, athletic build, dark curly hair, charming smile, wearing a light blue fitted shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    {"id": "kai", "name": "Kai Nakamura", "prompt": "photorealistic portrait of a handsome young Japanese man with light skin, lean build, modern styled black hair, thoughtful expression, wearing a yellow t-shirt, studio lighting, professional headshot, 4k, ultra detailed"},
    # WOMEN
    {"id": "sofia", "name": "Sofia Martinez", "prompt": "photorealistic portrait of a beautiful young Colombian woman with tan skin, fit body, long dark brown hair, stunning smile, wearing a pink crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "valentina", "name": "Valentina Cruz", "prompt": "photorealistic portrait of a beautiful young Colombian woman with bronze skin, athletic body, long dark wavy hair, confident expression, wearing a red fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "isabella", "name": "Isabella Rodriguez", "prompt": "photorealistic portrait of a gorgeous young Colombian woman with tan skin, fit body, long black hair, warm smile, wearing a purple crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "mariana", "name": "Mariana Lopez", "prompt": "photorealistic portrait of a beautiful young Colombian woman with medium skin, athletic body, long dark hair in ponytail, bright smile, wearing a pink halter top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "camila", "name": "Camila Gutierrez", "prompt": "photorealistic portrait of a beautiful young Colombian woman with bronze skin, fit body, long dark wavy hair, stunning expression, wearing a red fitted dress, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "ana", "name": "Ana Patricia", "prompt": "photorealistic portrait of a gorgeous young Colombian woman with tan skin, curvy body, long dark hair, confident smile, wearing an orange crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "rosa", "name": "Rosa Hernandez", "prompt": "photorealistic portrait of a beautiful young Dominican woman with tan skin, curvy body, long dark hair, warm expression, wearing a light blue top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "nayeli", "name": "Nayeli Fernandez", "prompt": "photorealistic portrait of a gorgeous young Puerto Rican woman with medium skin, fit body, long dark hair, bright smile, wearing a teal fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "mia", "name": "Mia Torres", "prompt": "photorealistic portrait of a beautiful young Puerto Rican woman with tan skin, fit body, long dark wavy hair, warm smile, wearing an orange crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "yolanda", "name": "Yolanda Vega", "prompt": "photorealistic portrait of a gorgeous young Dominican woman with bronze skin, athletic body, long dark hair, confident expression, wearing a yellow fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "carmen", "name": "Carmen Ruiz", "prompt": "photorealistic portrait of a beautiful young Mexican woman with medium skin, fit body, long dark hair, warm smile, wearing a magenta top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "leticia", "name": "Leticia Santos", "prompt": "photorealistic portrait of a beautiful young Brazilian woman with bronze skin, fit body, long dark wavy hair, stunning smile, wearing a green crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "luna", "name": "Luna Sanchez", "prompt": "photorealistic portrait of a gorgeous young Mexican woman with medium skin, athletic body, long black hair, bright expression, wearing a purple fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "naomi", "name": "Naomi Clarke", "prompt": "photorealistic portrait of a beautiful young Jamaican woman with deep dark skin, fit body, long dark natural hair, warm smile, wearing a teal crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "shanice", "name": "Shanice Brown", "prompt": "photorealistic portrait of a gorgeous young Jamaican woman with brown skin, curvy body, long dark hair, confident smile, wearing a pink fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "keisha", "name": "Keisha Campbell", "prompt": "photorealistic portrait of a beautiful young Jamaican woman with deep dark skin, athletic body, long braided hair, stunning expression, wearing an orange crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "aliyah", "name": "Aliyah Davis", "prompt": "photorealistic portrait of a gorgeous young black American woman with dark skin, athletic body, long natural hair, warm smile, wearing an orange fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "jasmine", "name": "Jasmine Okafor", "prompt": "photorealistic portrait of a beautiful young Nigerian woman with dark skin, athletic body, long natural hair, confident expression, wearing a light blue top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "emma", "name": "Emma Wilson", "prompt": "photorealistic portrait of a beautiful young American woman with light skin, fit body, long blonde hair, bright smile, wearing a purple fitted top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
    {"id": "hanna", "name": "Hanna Brooks", "prompt": "photorealistic portrait of a gorgeous young American woman with light skin, fit body, long auburn hair, warm expression, wearing a green crop top, studio lighting, professional headshot, 4k, ultra detailed, attractive"},
]

print(f"Loading Stable Diffusion pipeline on {device}...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Enable memory efficiency
if device == "mps":
    pipe.enable_attention_slicing()

print(f"Generating {len(characters)} character portraits...")
for i, char in enumerate(characters):
    outpath = os.path.join(OUTPUT_DIR, f"{char['id']}.png")
    if os.path.exists(outpath):
        print(f"[{i+1}/{len(characters)}] {char['name']} - already exists, skipping")
        continue
    
    print(f"[{i+1}/{len(characters)}] Generating {char['name']}...")
    try:
        image = pipe(
            char['prompt'],
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=768,
            generator=torch.Generator(device=device).manual_seed(42 + i)
        ).images[0]
        
        image.save(outpath)
        print(f"  ✅ Saved to {outpath}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\nDone! Portraits saved to {OUTPUT_DIR}")
