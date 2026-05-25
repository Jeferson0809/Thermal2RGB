#Pasar video partido en frames thermal a RGB
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

VIDEO_PATH = "../dataset/1535926_filtered.mp4"

START_FRAME = 45

MAX_FRAMES = 16

FRAME_STEP = 1

# ============================================================
# RESOLUTION
# ============================================================

WIDTH  = 1024
HEIGHT = 1024

SEED = 42

# ============================================================
# FOURIER
# ============================================================

FOURIER_SIGMA = 35

# ============================================================
# DIFFUSION
# ============================================================

NUM_INFERENCE_STEPS = 40

GUIDANCE_SCALE = 5.0

CONTROLNET_SCALE = 0.85

# ============================================================
# IP-ADAPTER
# ============================================================

IP_ADAPTER_SCALE = 0.65

# ============================================================
# VIDEO
# ============================================================

FPS = 8

# ============================================================
# PROMPTS
# ============================================================

PROMPT = (
    "wildlife trail camera photograph of a small kiwi "
    "walking in forest, "
    "realistic wildlife photography, "
    "natural anatomy, realistic fur, "
    "RAW DSLR wildlife image"
)

NEGATIVE_PROMPT = (
    "anime, cartoon, painting, illustration, cgi, "
    "cyberpunk, robot, futuristic, "
    "deformed anatomy, extra limbs, "
    "mutated animal, unrealistic fur, "
    "oversaturated, blurry, "
    "text, watermark"
)

# ============================================================
# OUTPUTS
# ============================================================

BASE_DIR = os.getcwd()

VIDEO_NAME = os.path.splitext(
    os.path.basename(VIDEO_PATH)
)[0]

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    VIDEO_NAME
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

CONSISTENT_DIR = os.path.join(
    OUTPUT_DIR,
    "rgb_consistent"
)

os.makedirs(THERMAL_DIR, exist_ok=True)
os.makedirs(FOURIER_DIR, exist_ok=True)
os.makedirs(RGB_DIR, exist_ok=True)
os.makedirs(CONSISTENT_DIR, exist_ok=True)

print("===================================")
print("OUTPUT DIR")
print("===================================")
print(OUTPUT_DIR)

# ============================================================
# SAVE CONFIG
# ============================================================

config_text = (
    f"VIDEO_PATH = {VIDEO_PATH}\n"
    f"START_FRAME = {START_FRAME}\n"
    f"MAX_FRAMES = {MAX_FRAMES}\n"
    f"FRAME_STEP = {FRAME_STEP}\n"
    f"\n"
    f"WIDTH = {WIDTH}\n"
    f"HEIGHT = {HEIGHT}\n"
    f"SEED = {SEED}\n"
    f"\n"
    f"FOURIER_SIGMA = {FOURIER_SIGMA}\n"
    f"\n"
    f"NUM_INFERENCE_STEPS = {NUM_INFERENCE_STEPS}\n"
    f"GUIDANCE_SCALE = {GUIDANCE_SCALE}\n"
    f"CONTROLNET_SCALE = {CONTROLNET_SCALE}\n"
    f"\n"
    f"IP_ADAPTER_SCALE = {IP_ADAPTER_SCALE}\n"
    f"\n"
    f"PROMPT = {PROMPT}\n"
    f"\n"
    f"NEGATIVE_PROMPT = {NEGATIVE_PROMPT}\n"
)

with open(
    os.path.join(OUTPUT_DIR, "config.txt"),
    "w"
) as f:

    f.write(config_text)


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

print("\n===================================")
print("DEVICE")
print("===================================")
print(DEVICE)

# ============================================================
# FOURIER EDGES (edgesfourier.py)
# ============================================================

def fourier_edges(
    thermal,
    sigma=35
):

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(thermal)

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # SMOOTH
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        mask,
        (9, 9),
        2
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # IFFT
    # --------------------------------------------------------

    inverse_shift = np.fft.ifftshift(filtered)

    inverse_fft = np.fft.ifft2(inverse_shift)

    edges = np.abs(inverse_fft)

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    edges = cv2.normalize(
        edges,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    edges = edges.astype(np.uint8)

    # --------------------------------------------------------
    # SOFTEN
    # --------------------------------------------------------

    edges = cv2.GaussianBlur(
        edges,
        (5, 5),
        0
    )

    return enhanced, edges

# ============================================================
# LOAD VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise Exception(
        f"No se pudo abrir: {VIDEO_PATH}"
    )

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    START_FRAME
)

control_images = []

frame_idx = 0
saved_count = 0

# ============================================================
# GENERATE FOURIER MAPS
# ============================================================

print("\n===================================")
print("GENERANDO FOURIER")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % FRAME_STEP != 0:

        frame_idx += 1
        continue

    if saved_count >= MAX_FRAMES:
        break

    print(
        f"\nFrame {saved_count}"
    )

    # --------------------------------------------------------
    # THERMAL
    # --------------------------------------------------------

    thermal = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    thermal = cv2.resize(
        thermal,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_CUBIC
    )

    thermal = cv2.fastNlMeansDenoising(
        thermal,
        None,
        6,
        7,
        21
    )

    # --------------------------------------------------------
    # FOURIER
    # --------------------------------------------------------

    enhanced, edges = fourier_edges(
        thermal,
        sigma=FOURIER_SIGMA
    )

    # --------------------------------------------------------
    # CONTROL IMAGE
    # --------------------------------------------------------

    control_rgb = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2RGB
    )

    control_pil = Image.fromarray(
        control_rgb
    )

    control_images.append(control_pil)

    # --------------------------------------------------------
    # SAVE INTERMEDIATE
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            THERMAL_DIR,
            f"thermal_{saved_count:05d}.png"
        ),
        enhanced
    )

    cv2.imwrite(
        os.path.join(
            FOURIER_DIR,
            f"fourier_{saved_count:05d}.png"
        ),
        edges
    )

    saved_count += 1
    frame_idx += 1

cap.release()

print("\nTotal frames:", saved_count)

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
# LOAD PIPELINE
# ============================================================

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=DTYPE,
    use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config
)

pipe.to(DEVICE)

# IMPORTANT:
# attention slicing rompe IP-Adapter

pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

print("Pipeline cargado.")

# ============================================================
# FIRST PASS
# ============================================================

print("\n===================================")
print("GENERANDO RGB INICIAL")
print("===================================")

generator = torch.Generator(
    device=DEVICE
).manual_seed(SEED)

for i, control_image in enumerate(control_images):

    print(f"\nFrame {i+1}/{saved_count}")

    result = pipe(
        prompt=PROMPT,

        negative_prompt=NEGATIVE_PROMPT,

        image=control_image,

        num_inference_steps=NUM_INFERENCE_STEPS,

        guidance_scale=GUIDANCE_SCALE,

        controlnet_conditioning_scale=CONTROLNET_SCALE,

        generator=generator
    ).images[0]

    rgb_path = os.path.join(
        RGB_DIR,
        f"rgb_{i:05d}.png"
    )

    result.save(rgb_path)

    print("RGB guardado:", rgb_path)

# ============================================================
# USER SELECTS BEST FRAME
# ============================================================

print("\n===================================")
print("FRAMES DISPONIBLES")
print("===================================")

for i in range(saved_count):

    print(f"[{i}] rgb_{i:05d}.png")

print("\n")

selected_idx = int(
    input(
        "Selecciona el mejor frame: "
    )
)

reference_filename = (
    f"rgb_{selected_idx:05d}.png"
)

reference_path = os.path.join(
    RGB_DIR,
    reference_filename
)

if not os.path.exists(reference_path):

    raise Exception(
        f"No existe: {reference_path}"
    )

reference_image = Image.open(
    reference_path
).convert("RGB")

print("\n===================================")
print("FRAME REFERENCIA")
print("===================================")

print(reference_filename)

# ============================================================
# LOAD IP-ADAPTER
# ============================================================

print("\n===================================")
print("CARGANDO IP-ADAPTER")
print("===================================")

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

pipe.set_ip_adapter_scale(
    IP_ADAPTER_SCALE
)

print("IP-Adapter cargado.")

# ============================================================
# SECOND PASS
# ============================================================

print("\n===================================")
print("GENERANDO FRAMES COHERENTES")
print("===================================")

for i, control_image in enumerate(control_images):

    print(f"\nFrame coherente {i+1}/{saved_count}")

    result = pipe(
        prompt=PROMPT,

        negative_prompt=NEGATIVE_PROMPT,

        image=control_image,

        ip_adapter_image=reference_image,

        num_inference_steps=NUM_INFERENCE_STEPS,

        guidance_scale=GUIDANCE_SCALE,

        controlnet_conditioning_scale=CONTROLNET_SCALE,

        generator=generator
    ).images[0]

    save_path = os.path.join(
        CONSISTENT_DIR,
        f"consistent_{i:05d}.png"
    )

    result.save(save_path)

    print("Guardado:", save_path)

# ============================================================
# CREATE VIDEO
# ============================================================

print("\n===================================")
print("CREANDO VIDEO")
print("===================================")

frames = sorted(
    os.listdir(CONSISTENT_DIR)
)

first = cv2.imread(
    os.path.join(
        CONSISTENT_DIR,
        frames[0]
    )
)

h, w, _ = first.shape

video_path = os.path.join(
    OUTPUT_DIR,
    "consistent_video.mp4"
)

video = cv2.VideoWriter(
    video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    (w, h)
)

for f in frames:

    frame = cv2.imread(
        os.path.join(
            CONSISTENT_DIR,
            f
        )
    )

    video.write(frame)

video.release()

# ============================================================
# DONE
# ============================================================

print("\n===================================")
print("TERMINADO")
print("===================================")

print("Thermal:", THERMAL_DIR)
print("Fourier:", FOURIER_DIR)
print("RGB inicial:", RGB_DIR)
print("RGB coherente:", CONSISTENT_DIR)
print("Video:", video_path)
