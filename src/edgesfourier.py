import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

FRAMES_DIR = "video_frames" #Carpeta con los frames

SELECTED_FRAMES = [ #Frame elegido para sacar bordes
    "frame_00088.jpg"
]

OUTPUT_DIR = "final_fourier_edges"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# PROCESS
# ============================================================

for frame_name in SELECTED_FRAMES:

    print(f"\nProcessing {frame_name}")

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    frame_path = os.path.join(
        FRAMES_DIR,
        frame_name
    )

    img = cv2.imread(
        frame_path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        print(f"Could not load {frame_name}")
        continue

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    img = cv2.resize(
        img,
        (512,512)
    )

    # ========================================================
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #  Mejora del contraste
    # ========================================================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(img)

    # ========================================================
    # 2. OTSU THRESHOLD
    # Threshold automatico para hacer la segmentación
    # ========================================================

    otsu_value, mask = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print("OTSU Threshold:", otsu_value)

    # ========================================================
    # 3. GAUSSIAN SMOOTH
    # Suavizar la imagen termica.
    # ========================================================

    smooth_mask = cv2.GaussianBlur(
        mask,
        (9,9),
        2
    )

    # ========================================================
    # 4. FFT
    # FOURIER
    # ========================================================

    fft = np.fft.fft2(
        smooth_mask
    )

    fft_shift = np.fft.fftshift(
        fft
    )

    # ========================================================
    # FFT VISUALIZATION
    # ========================================================

    fft_visual = np.log(
        np.abs(fft_shift) + 1
    )

    # ========================================================
    # 5. GAUSSIAN HIGH PASS FILTER
    # Solo dejamos pasar las altas frecuencias que son bordes, que es lo que necesitamos
    # ========================================================

    rows, cols = smooth_mask.shape

    x = np.linspace(
        -cols//2,
        cols//2,
        cols
    )

    y = np.linspace(
        -rows//2,
        rows//2,
        rows
    )

    X, Y = np.meshgrid(
        x,
        y
    )

    D = np.sqrt(
        X**2 + Y**2
    )

    sigma = 35

    low_pass = np.exp(
        -(D**2) / (2 * sigma**2)
    )

    high_pass = 1 - low_pass

    # ========================================================
    # 6. APPLY HIGH PASS FILTER
    # ========================================================

    filtered = fft_shift * high_pass

    # ========================================================
    # FILTERED FFT VISUALIZATION
    # ========================================================

    filtered_visual = np.log(
        np.abs(filtered) + 1
    )

    # ========================================================
    # 7. INVERSE FFT
    # ========================================================

    inverse_shift = np.fft.ifftshift(
        filtered
    )

    inverse_fft = np.fft.ifft2(
        inverse_shift
    )

    edges = np.abs(
        inverse_fft
    )

    # ========================================================
    # NORMALIZE FOR DISPLAY
    # ========================================================

    edges = cv2.normalize(
        edges,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    edges = edges.astype(
        np.uint8
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_name = frame_name.replace(
        ".jpg",
        "_edge.png"
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        save_name
    )

    cv2.imwrite(
        save_path,
        edges
    )

    print(f"Saved: {save_name}")

    # ========================================================
    # DISPLAY PIPELINE
    # ========================================================

    images = [
        img,
        enhanced,
        mask,
        smooth_mask,
        fft_visual,
        high_pass,
        filtered_visual,
        edges
    ]

    titles = [
        "1. Original",
        "2. CLAHE",
        "3. OTSU Mask",
        "4. Smoothed Mask",
        "5. FFT",
        "6. High Pass Filter",
        "7. Filtered FFT",
        "8. Fourier Edges"
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18,10)
    )

    fig.suptitle(
        frame_name,
        fontsize=18
    )

    for ax, image, title in zip(
        axes.flat,
        images,
        titles
    ):

        ax.imshow(
            image,
            cmap="gray"
        )

        ax.set_title(
            title,
            fontsize=10
        )

        ax.axis("off")

    plt.tight_layout()

    plt.show()