#Pasar imagen termica.npy a RGB
import os
import cv2
import torch
import numpy as np

from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    UniPCMultistepScheduler
)

# ============================================================
# CONFIG
# ============================================================

NPY_PATH = "../imagenes/FLIR0005_temp.npy"

WIDTH  = 1024
HEIGHT = 1024

SEED = 42

FOURIER_SIGMA = 35

NUM_INFERENCE_STEPS = 45
CONTROLNET_SCALE    = 0.85
GUIDANCE_SCALE      = 10.5  # Subir para que el prompt domine más

# ============================================================
# PROMPTS
# ============================================================

SEED = 256  # o probar 7, 512, 1337

PROMPT = (
    "RAW photo, one man standing alone, full body, "
    "back to camera, rear view, seen from behind, "
    "arms at sides, standing upright, "
    "casual clothing, jeans and jacket, "
    "outdoor urban environment, street background, "
    "eye level camera angle, "  # <-- evita vista aérea
    "photorealistic, sharp focus, DSLR, 35mm, 8k"
)

NEGATIVE_PROMPT = (
    "multiple people, crowd, group, two people, "
    "aerial view, bird's eye view, top down, drone, "  # <-- clave
    "facing camera, front view, face visible, "
    "painting, cartoon, anime, illustration, "
    "blurry, neon, silhouette, monochrome, "
    "deformed, bad anatomy, text, watermark"
)
# ============================================================
# OUTPUT DIRS
# ============================================================

BASE_DIR = os.getcwd()

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "thermal_person_output"
)

THERMAL_DIR = os.path.join(
    OUTPUT_DIR,
    "thermal"
)

FOURIER_DIR = os.path.join(
    OUTPUT_DIR,
    "fourier"
)

RGB_DIR = os.path.join(
    OUTPUT_DIR,
    "rgb"
)

os.makedirs(THERMAL_DIR, exist_ok=True)
os.makedirs(FOURIER_DIR, exist_ok=True)
os.makedirs(RGB_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

print("\nDEVICE:", DEVICE)

# ============================================================
# FOURIER FUNCTION
# ============================================================

def fourier_edges(
    thermal,
    sigma=35
):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(thermal)

    _, mask = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    smooth = cv2.GaussianBlur(
        mask,
        (9, 9),
        2
    )

    fft = np.fft.fft2(smooth)

    fft_shift = np.fft.fftshift(fft)

    rows, cols = smooth.shape

    x = np.linspace(
        -cols // 2,
        cols // 2,
        cols
    )

    y = np.linspace(
        -rows // 2,
        rows // 2,
        rows
    )

    X, Y = np.meshgrid(x, y)

    D = np.sqrt(X**2 + Y**2)

    low_pass = np.exp(
        -(D**2) / (2 * sigma**2)
    )

    high_pass = 1 - low_pass

    filtered = fft_shift * high_pass

    inverse_shift = np.fft.ifftshift(filtered)

    inverse_fft = np.fft.ifft2(inverse_shift)

    edges = np.abs(inverse_fft)

    edges = cv2.normalize(
        edges,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    edges = edges.astype(np.uint8)

    edges = cv2.GaussianBlur(
        edges,
        (5, 5),
        0
    )

    return enhanced, edges

# ============================================================
# LOAD NPY
# ============================================================

print("\n===================================")
print("CARGANDO NPY")
print("===================================")

thermal = np.load(NPY_PATH)

print("Shape original:", thermal.shape)
print("dtype:", thermal.dtype)

# ============================================================
# NORMALIZE
# ============================================================

thermal = thermal.astype(np.float32)

thermal = cv2.normalize(
    thermal,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

thermal = thermal.astype(np.uint8)

# ============================================================
# RESIZE
# ============================================================

thermal = cv2.resize(
    thermal,
    (WIDTH, HEIGHT),
    interpolation=cv2.INTER_CUBIC
)

# ============================================================
# DENOISE
# ============================================================

thermal = cv2.bilateralFilter(
    thermal,
    9,
    75,
    75
)

# ============================================================
# FOURIER
# ============================================================

print("\n===================================")
print("GENERANDO FOURIER")
print("===================================")

enhanced, edges = fourier_edges(
    thermal,
    sigma=FOURIER_SIGMA
)

# ============================================================
# SAVE THERMAL / FOURIER
# ============================================================

thermal_path = os.path.join(
    THERMAL_DIR,
    "thermal.png"
)

fourier_path = os.path.join(
    FOURIER_DIR,
    "fourier.png"
)

cv2.imwrite(
    thermal_path,
    enhanced
)

cv2.imwrite(
    fourier_path,
    edges
)

print("Thermal:", thermal_path)
print("Fourier:", fourier_path)

# ============================================================
# CONTROL IMAGE
# ============================================================

control_rgb = cv2.cvtColor(
    edges,
    cv2.COLOR_GRAY2RGB
)

control_pil = Image.fromarray(
    control_rgb
)

# ============================================================
# LOAD CONTROLNET
# ============================================================

print("\n===================================")
print("CARGANDO CONTROLNET")
print("===================================")

controlnet = ControlNetModel.from_pretrained(
    "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
    torch_dtype=DTYPE
)

# ============================================================
# LOAD VAE
# ============================================================

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=DTYPE
)

# ============================================================
# PIPELINE
# ============================================================

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=DTYPE,
    use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe.to(DEVICE)

pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# ============================================================
# IMPORTANTE
# ============================================================

pipe.unload_ip_adapter()

# ============================================================
# GENERATOR
# ============================================================

generator = torch.Generator(
    device=DEVICE
).manual_seed(SEED)

# ============================================================
# GENERACION RGB
# ============================================================

print("\n===================================")
print("GENERANDO RGB")
print("===================================")

result = pipe(
    prompt=PROMPT,

    negative_prompt=NEGATIVE_PROMPT,

    image=control_pil,

    num_inference_steps=NUM_INFERENCE_STEPS,

    guidance_scale=GUIDANCE_SCALE,

    controlnet_conditioning_scale=CONTROLNET_SCALE,

    generator=generator
).images[0]

# ============================================================
# SAVE RGB
# ============================================================

rgb_path = os.path.join(
    RGB_DIR,
    "rgb.png"
)

result.save(rgb_path)

print("RGB:", rgb_path)

# ============================================================
# DONE
# ============================================================

print("\n===================================")
print("TERMINADO")
print("===================================")
