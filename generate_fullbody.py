#!/usr/bin/env python3
"""Regenerate all 30 portraits with FULL BODY and more realistic prompts."""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os

device = "mps"
OUTPUT_DIR = os.path.expanduser("~/.openclaw/workspace/ai-world-3d/public/portraits")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Realistic photo prompts — full body, natural lighting, slight imperfections
characters = [
    # MEN
    {"id": "omar", "prompt": "full body photo of a handsome young Jamaican black man, athletic build, short dreads, dark skin, confident stance, wearing cyan fitted t-shirt and dark jeans, standing casually, natural outdoor lighting, shot on Canon EOS R5, 85mm lens, shallow depth of field, slight skin texture, real person not AI, candid photography, 4k"},
    {"id": "tony", "prompt": "full body photo of an attractive young Jamaican man, brown skin, short fade haircut, charismatic smile, wearing orange fitted shirt and black pants, standing with arms crossed, natural golden hour lighting, shot on Canon EOS R5, 85mm lens, shallow depth of field, real person not AI, candid photography, 4k"},
    {"id": "marcus", "prompt": "full body photo of a handsome black American man, dark skin, broad shoulders, clean shaved head, serious expression, wearing navy blue dress shirt tucked in, standing tall, professional office lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "kwame", "prompt": "full body photo of a handsome young Ghanaian man, deep dark skin, athletic build, short curly hair, warm smile, wearing orange polo shirt and khakis, standing relaxed, natural daylight, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "deon", "prompt": "full body photo of a handsome mixed-race man, brown skin, broad build, neat beard, confident pose, wearing purple dress shirt and dark trousers, standing casually, studio lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "carlos", "prompt": "full body photo of a handsome young Colombian man, tan skin, athletic build, dark wavy hair, warm expression, wearing green fitted t-shirt and jeans, standing with one hand in pocket, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "miguel", "prompt": "full body photo of a handsome young Mexican man, medium skin, lean build, black hair styled neatly, friendly smile, wearing teal polo shirt and chinos, standing relaxed, outdoor lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "jin", "prompt": "full body photo of a handsome young Korean man, light skin, lean build, modern styled black hair, warm expression, wearing pink fitted shirt and dark jeans, standing casually, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "rafael", "prompt": "full body photo of a handsome young Brazilian man, bronze skin, athletic build, dark curly hair, charming smile, wearing light blue fitted shirt and white pants, standing with arms relaxed, golden hour lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "kai", "prompt": "full body photo of a handsome young Japanese man, light skin, lean build, modern styled black hair, thoughtful expression, wearing yellow t-shirt and dark chinos, standing casually, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    # WOMEN
    {"id": "sofia", "prompt": "full body photo of a beautiful young Colombian woman, tan skin, fit body, long dark brown hair flowing, stunning smile, wearing pink crop top and high-waisted jeans, standing confidently, natural golden hour lighting, shot on Canon EOS R5, 85mm lens, shallow depth of field, real person not AI, candid photography, 4k"},
    {"id": "valentina", "prompt": "full body photo of a beautiful young Colombian woman, bronze skin, athletic body, long dark wavy hair, confident pose, wearing red fitted dress, standing elegantly, studio lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "isabella", "prompt": "full body photo of a gorgeous young Colombian woman, tan skin, fit body, long black hair, warm smile, wearing purple crop top and skirt, standing casually, natural outdoor lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "mariana", "prompt": "full body photo of a beautiful young Colombian woman, medium skin, athletic body, long dark hair in ponytail, bright smile, wearing pink halter top and jeans, standing with hand on hip, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "camila", "prompt": "full body photo of a beautiful young Colombian woman, bronze skin, fit body, long dark wavy hair, stunning expression, wearing red fitted top and black pants, standing confidently, golden hour lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "ana", "prompt": "full body photo of a gorgeous young Colombian woman, tan skin, curvy body, long dark hair, confident smile, wearing orange crop top and high-waisted shorts, standing casually, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "rosa", "prompt": "full body photo of a beautiful young Dominican woman, tan skin, curvy body, long dark hair, warm expression, wearing light blue sundress, standing elegantly, tropical outdoor lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "nayeli", "prompt": "full body photo of a gorgeous young Puerto Rican woman, medium skin, fit body, long dark hair, bright smile, wearing teal fitted top and white skirt, standing confidently, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "mia", "prompt": "full body photo of a beautiful young Puerto Rican woman, tan skin, fit body, long dark wavy hair, warm smile, wearing orange crop top and jeans, standing casually, golden hour lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "yolanda", "prompt": "full body photo of a gorgeous young Dominican woman, bronze skin, athletic body, long dark hair, confident expression, wearing yellow fitted dress, standing elegantly, studio lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "carmen", "prompt": "full body photo of a beautiful young Mexican woman, medium skin, fit body, long dark hair, warm smile, wearing magenta top and denim skirt, standing with hand in pocket, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "leticia", "prompt": "full body photo of a beautiful young Brazilian woman, bronze skin, fit body, long dark wavy hair, stunning smile, wearing green crop top and white pants, standing confidently, beach outdoor lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "luna", "prompt": "full body photo of a gorgeous young Mexican woman, medium skin, athletic body, long black hair, bright expression, wearing purple fitted top and jeans, standing casually, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "naomi", "prompt": "full body photo of a beautiful young Jamaican woman, deep dark skin, fit body, long dark natural hair, warm smile, wearing teal crop top and high-waisted jeans, standing confidently, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "shanice", "prompt": "full body photo of a gorgeous young Jamaican woman, brown skin, curvy body, long dark hair, confident smile, wearing pink fitted dress, standing elegantly, studio lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "keisha", "prompt": "full body photo of a beautiful young Jamaican woman, deep dark skin, athletic body, long braided hair, stunning expression, wearing orange fitted top and skirt, standing with arms relaxed, golden hour lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "aliyah", "prompt": "full body photo of a gorgeous young black American woman, dark skin, athletic body, long natural hair, warm smile, wearing orange fitted top and dark jeans, standing confidently, natural lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "jasmine", "prompt": "full body photo of a beautiful young Nigerian woman, dark skin, athletic body, long natural hair, confident expression, wearing light blue top and patterned skirt, standing elegantly, studio lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "emma", "prompt": "full body photo of a beautiful young American woman, light skin, fit body, long blonde hair, bright smile, wearing purple fitted top and white shorts, standing casually, natural outdoor lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
    {"id": "hanna", "prompt": "full body photo of a gorgeous young American woman, light skin, fit body, long auburn hair, warm expression, wearing green crop top and jeans, standing confidently, golden hour lighting, shot on Canon EOS R5, 85mm lens, real person not AI, candid photography, 4k"},
]

print(f"Loading pipeline on {device}...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

print(f"Regenerating {len(characters)} FULL BODY portraits...")
for i, char in enumerate(characters):
    outpath = os.path.join(OUTPUT_DIR, f"{char['id']}.png")
    print(f"[{i+1}/{len(characters)}] {char['id']}...")
    try:
        image = pipe(
            char['prompt'],
            num_inference_steps=30,
            guidance_scale=7.5,
            width=512,
            height=768,
            generator=torch.Generator(device=device).manual_seed(200 + i)
        ).images[0]
        image.save(outpath)
        print(f"  ✅ Done")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\nAll 30 full-body portraits regenerated!")
