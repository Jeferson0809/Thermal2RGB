<div align="center">

# Thermal2RGB - Fourier
<img width="1200" height="500" alt="THERMAL" src="assets/THERMAL.png" />

### Generación de imágenes RGB a partir de videos térmicos utilizando Fourier.


<br>

[Jeferson Acevedo](https://github.com/Jeferson0809) • [Brayan Quintero](https://github.com/BrayanQuintero123)

---

</div>

La generación de imágenes RGB a partir de cámaras térmicas es un problema relevante en visión por computador, especialmente en escenarios nocturnos, vigilancia, wildlife monitoring y percepción multimodal.

Este proyecto propone un pipeline de **Thermal-to-RGB** basado en extracción de bordes mediante Fourier y modelos generativos basados en difusión. La información estructural de las imágenes térmicas es utilizada como guía para modelos SDXL condicionados con ControlNet, permitiendo generar imágenes RGB coherentes y visualmente realistas.

Además, se incorpora consistencia temporal utilizando IP-Adapter para mejorar la estabilidad entre frames consecutivos en secuencias de video.

> **Objetivo:** Utilizar información estructural obtenida en el dominio de Fourier para guiar modelos generativos y transformar videos térmicos en secuencias RGB coherentes.


---

# Arquitectura

<img width="1500" height="700" alt="THERMAL" src="assets/arquitectura.jpeg" />

---

# Pipeline

```text
Thermal Video
      ↓
CLAHE Enhancement
      ↓
OTSU Segmentation
      ↓
Gaussian Smoothing
      ↓
FFT Transformation
      ↓
High-Pass Fourier Filter
      ↓
Fourier Edge Extraction
      ↓
ControlNet Conditioning
      ↓
SDXL RGB Generation
      ↓
IP-Adapter Temporal Consistency
      ↓
Final RGB Video

```

# Resultados

<div align="center">

| Fourier Edge Extraction | RGB Generation |
|---|---|
| <img src="assets/resultadofourier.gif" width="450"> | <img src="assets/resultadoRGB.gif" width="450"> |

</div>

---



---

# Estructura del repositorio

- `dataset/` — Videos térmicos utilizados para pruebas y generación obtenidos de https://lila.science/datasets/new-zealand-wildlife-thermal-imaging/
- `assets/` — GIFs y visualizaciones de resultados.
- `outputs/` — Resultados generados por el pipeline.
- `src/` — Implementación principal del pipeline Thermal-to-RGB.
  - `edgesfourier.py` — Extracción de bordes mediante Fourier.
  - `thermal2rgb.py` — Pipeline completo de generación RGB para videos.
  - `thermal2rgb2.py` — Pipeline completo de generación RGB para imagenes.
- `README.md` — Documentación principal del proyecto.

---

# Fourier Edge Extraction

La etapa de preprocessing utiliza técnicas de procesamiento de imágenes para extraer información estructural desde imágenes térmicas.

## Etapas principales

1. CLAHE para mejora de contraste.
2. Segmentación automática mediante OTSU.
3. Suavizado Gaussiano.
4. Transformada rápida de Fourier (FFT).
5. Filtro High-Pass Gaussiano.
6. Reconstrucción mediante Inverse FFT.
7. Normalización y generación de mapas de bordes.

Estos mapas son utilizados posteriormente como condición estructural para ControlNet.

---

# Modelos utilizados

## Generación RGB
- SDXL
- RealVisXL V4.0

## ControlNet
- SoftEdge DexiNed

## Consistencia temporal
- IP-Adapter SDXL

---

# Instalación

Clonar el repositorio:

```bash
git clone https://github.com/Jeferson0809/Thermal2RGB.git

cd Thermal2RGB
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Uso

## Extracción de bordes Fourier

```bash
python src/edgesfourier.py
```

---

## Generación Thermal-to-RGB

```bash
python src/thermal2rgb.py
```

---

# Salidas generadas

El pipeline produce automáticamente:

```bash
thermal/
fourier/
rgb/
rgb_consistent/
consistent_video.mp4
```

