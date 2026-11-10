"""
Demo: detectar y dibujar el contorno del cordon (forma de "L") en imagenes de microscopio.

Las fotos son de tipos muy distintos, asi que cada tipo tiene un PERFIL (ImageProfile) registrado
en PROFILES: nombre, descripcion, carpeta, funcion de segmentacion y sus parametros.
  Muestra_SD -> muestra (probeta) vista desde arriba (FOTOS/CS): cordon oscuro sobre fondo claro.
  Corte_SD   -> corte de la soldadura con colores reales (FOTOS/CC): cordon distinguible por textura.

Pipeline comun:
  1. Cargar imagen y reducirla (el contorno se escala de vuelta al original).
  2. Segmentar con la funcion del perfil -> mascara binaria del cordon.
  3. Mayor contorno externo, simplificado y reescalado.
  4. Dibujar contorno sobre la imagen original y calcular el area en pixeles.

Para anadir un tipo nuevo: escribir `segment_xxx(small, p, steps)` y registrarlo con register().

Uso:
    python demo_contour.py                         # perfil Muestra_SD: FOTOS/CS/*.tif
    python demo_contour.py ruta/imagen.tif         # perfil segun la carpeta de la imagen
    python demo_contour.py --profile Corte_SD a.tif b.tif  # forzar perfil
    python demo_contour.py --list-profiles         # ver perfiles disponibles
    python demo_contour.py --steps                 # guarda ademas imagenes de cada paso
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

import scale

def desktop_dir():
    """Escritorio del usuario (respeta la redireccion a OneDrive en Windows)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0:  # CSIDL_DESKTOPDIRECTORY
                return Path(buf.value)
        except Exception:
            pass
    return Path.home() / "Desktop"


BASE_DIR = Path(__file__).parent
FOTOS_DIR = BASE_DIR / "FOTOS"                  # solo como carpeta inicial sugerida (puede no existir)
OUTPUT_DIR = desktop_dir() / "analysis_results"     # se crea automaticamente

BLACK_LEVEL = 25      # por debajo de este valor se considera marco negro del microscopio


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    return img


def valid_region_mask(gray):
    """Mascara de pixeles utiles: excluye el marco negro (conectado a los bordes)."""
    black = (gray < BLACK_LEVEL).astype(np.uint8)
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    h, w = gray.shape
    border_black = np.zeros_like(black)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touches and area > 0.002 * h * w:
            border_black[labels == i] = 1
    border_black = cv2.dilate(border_black, np.ones((15, 15), np.uint8))
    return (border_black == 0)


def _ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _normalized_blur(x, weight, sigma):
    """Suavizado gaussiano que solo promedia pixeles con peso > 0 (convolucion normalizada)."""
    return cv2.GaussianBlur(x * weight, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(weight, (0, 0), sigma), 1e-4)


def _largest_component(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)
    return np.where(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)


def _fill_external(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def clean_mask(binary):
    """Erosion/apertura para quitar ruido, cierre para unir, y quedarse con el mayor componente."""
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask = cv2.erode(binary, k_small, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_big)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_big, iterations=2)
    mask = cv2.dilate(mask, k_small, iterations=2)  # compensa la erosion inicial

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Rellenar huecos internos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def refine_bright_edges(gray, valid, core, ratio_thr, band_width):
    """Recupera zonas del cordon con mas reflectividad (mas claras que el resto del cordon).

    El umbral Otsu deja fuera los lobulos/bordes claros del cordon. Aqui se estima el fondo
    local solo con pixeles lejos del cordon (convolucion normalizada) y se marca como cordon
    todo lo que es claramente mas oscuro que ese fondo (cociente < ratio_thr), pero solo
    en una banda estrecha alrededor del nucleo para no capturar manchas oscuras del fondo.
    """
    outside = ((cv2.dilate(core, np.ones((81, 81), np.uint8)) == 0) & valid).astype(np.float32)
    fine = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32)
    sigma = 60
    bg = cv2.GaussianBlur(fine * outside, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(outside, (0, 0), sigma), 1e-3)
    ratio = fine / np.maximum(bg, 1.0)

    band = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_width, band_width)))
    weak = (((ratio < ratio_thr) & (band > 0) & valid) * 255).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    mask = cv2.bitwise_or(weak, core)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled, ratio


def segment_dark(small, p, steps=None):
    """Cordon mas oscuro que el fondo, con iluminacion muy desigual (perfil Muestra_SD).

    El brillo del fondo varia mucho dentro de una misma foto (vineteo, reflejos), asi que no se usa
    ningun umbral de brillo absoluto. Se trabaja con el logaritmo del gris (el cordon es un cociente
    de brillo respecto al fondo local, no una resta) y se mide siempre contra el fondo de ESA zona:
      1. Semilla: filtro paso-banda (gris - fondo suavizado) en la zona segura (lejos del marco y del
         borde, donde vive el vineteo); el percentil mas oscuro da el nucleo del cordon.
      2. Refinado (iterativo): el fondo se re-estima solo con pixeles lejos del cordon actual y el
         borde se fija a una fraccion del contraste cordon/fondo de la propia imagen. Histeresis:
         umbral flojo, pero solo regiones conectadas a pixeles claramente de cordon.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    valid = valid_region_mask(gray)
    valid_f = valid.astype(np.float32)
    sigma_bg = p["bg_sigma"]

    safe = cv2.erode(valid.astype(np.uint8), _ellipse(p["safe"])) > 0
    m = p["safe"] // 2
    safe[:m, :] = safe[-m:, :] = False
    safe[:, :m] = safe[:, -m:] = False

    fine = _normalized_blur(np.log(np.maximum(gray.astype(np.float32), 1)), valid_f, p["fine_sigma"])
    bandpass = fine - _normalized_blur(fine, valid_f, sigma_bg)

    seed = np.where((bandpass < np.percentile(bandpass[safe], p["seed_pct"])) & safe, 255, 0).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, _ellipse(9))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, _ellipse(21), iterations=2)
    mask = _fill_external(_largest_component(seed))
    core = mask

    contrast = bandpass
    for _ in range(p["iters"]):
        far = (cv2.dilate(mask, _ellipse(p["excl"])) == 0) & valid
        inside = cv2.erode(mask, _ellipse(p["erode"])) > 0
        if not inside.any() or far.sum() < 500:
            break
        contrast = fine - _normalized_blur(fine, far.astype(np.float32), sigma_bg)
        # Niveles de cordon y de fondo LOCALES: un tramo del cordon mas claro (mas debil) no queda excluido
        # por el contraste de otro tramo mas oscuro.
        dark_level = _normalized_blur(contrast, inside.astype(np.float32), p["level_sigma"])
        bg_level = _normalized_blur(contrast, far.astype(np.float32), p["level_sigma"])
        gap = np.minimum(dark_level - bg_level, -p["min_contrast"])
        strong = bg_level + p["frac_strong"] * gap
        weak = bg_level + p["frac_weak"] * gap

        band = (cv2.dilate(mask, _ellipse(p["band"])) > 0) & valid
        strong_px = cv2.morphologyEx(np.where(band & (contrast < strong), 255, 0).astype(np.uint8),
                                     cv2.MORPH_OPEN, _ellipse(9))
        weak_px = np.where(band & (contrast < weak), 255, 0).astype(np.uint8)
        n, labels, _, _ = cv2.connectedComponentsWithStats(weak_px, connectivity=8)
        keep = np.unique(labels[strong_px > 0])
        grown = np.isin(labels, keep[keep > 0]).astype(np.uint8) * 255
        grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, _ellipse(9))
        grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, _ellipse(21))
        grown = _largest_component(grown)
        if not grown.any():
            break
        mask = _fill_external(grown)

    if steps is not None:
        steps.update({"1_paso_banda": np.clip((bandpass + 0.25) / 0.5 * 255, 0, 255).astype(np.uint8),
                      "2_semilla": core, "3_contraste_fondo": np.clip((contrast + 0.25) / 0.5 * 255, 0, 255).astype(np.uint8),
                      "4_mascara": mask})
    return mask


def horizontal_grain_score(small, sigma):
    """Coherencia de grano horizontal por pixel (tensor de estructura): +1 = estrias horizontales.

    El metal base tiene el grano horizontal (gradientes casi solo verticales); la soldadura y su
    zona de transicion no lo tienen (valores cercanos a 0).
    """
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    return (jyy - jxx) / (jyy + jxx + 1e-6)


def grow_to_interface(small, core, p):
    """Extiende el nucleo texturado hasta la interfaz real metal base / soldadura.

    Entre el nucleo (muy texturado) y el metal base hay una banda oscura y poco texturada que
    tambien es soldadura. Se distingue del metal base por no tener grano horizontal: el umbral
    es el punto medio entre la mediana del grano dentro del nucleo y la del metal base lejano.
    Solo se busca en una banda alrededor del nucleo para no escaparse por el metal base.
    """
    score = horizontal_grain_score(small, p["grain_sigma"])
    inside = cv2.erode(core, np.ones((31, 31), np.uint8)) > 0
    far = cv2.dilate(core, np.ones((151, 151), np.uint8)) == 0
    if not inside.any() or not far.any():
        return core, score
    lo_val, hi_val = np.median(score[inside]), np.median(score[far])
    thr = lo_val + p["grain_frac"] * (hi_val - lo_val)

    band = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["grain_band"], p["grain_band"]))) > 0
    weak = ((score < thr) & band).astype(np.uint8) * 255
    weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["grain_open"], p["grain_open"])))

    mask = cv2.bitwise_or(weak, core)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    mask = np.where(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled, score


def segment_texture(small, p, steps=None):
    """Cordon distinguible por textura + grano (perfil Corte_SD, corte con colores reales).

    1) Nucleo: el cordon esta mucho mas texturado que el metal base (desviacion tipica local
       del gris, suavizada y separada con Otsu).
    2) Interfaz: se extiende el nucleo hasta donde acaba la banda de transicion, que se
       distingue del metal base por no tener grano horizontal.
    """
    win = p["tex_win"]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (win, win))
    local_std = np.sqrt(np.maximum(cv2.blur(gray * gray, (win, win)) - mean ** 2, 0))
    texture = cv2.GaussianBlur(local_std, (0, 0), p["tex_sigma"])
    tex8 = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary = cv2.threshold(tex8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core = clean_mask(binary)  # apertura/cierre, mayor componente y relleno de huecos
    valid = valid_region_mask(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    # Las cavidades/defectos muy oscuros tienen poca textura: se anexan si estan junto al nucleo.
    core_dark, _ = refine_bright_edges(gray.astype(np.uint8), valid, core, p["dark_ratio"], p["dark_band"])
    mask, score = grow_to_interface(small, core_dark, p)

    if steps is not None:
        steps.update({"1_textura": tex8, "2_umbral_textura": binary, "3_nucleo": core_dark,
                      "4_grano_horizontal": np.clip((score + 1) / 2 * 255, 0, 255).astype(np.uint8),
                      "5_mascara": mask})
    return mask


# ============================================================================
# REGISTRO DE PERFILES (tipos de foto)
# ============================================================================

@dataclass(frozen=True)
class ImageProfile:
    key: str                      # nombre del perfil (tambien subcarpeta de resultados)
    folder: str                   # carpeta de entrada dentro de FOTOS/
    label: str                    # nombre legible (para un futuro selector en la interfaz)
    description: str
    segment: Callable             # (small_bgr, params, steps) -> mascara uint8 0/255
    params: dict = field(default_factory=dict)

    @property
    def input_dir(self):
        return FOTOS_DIR / self.folder

    @property
    def output_dir(self):
        return OUTPUT_DIR / self.key


PROFILES: dict = {}
DEFAULT_PROFILE = "Muestra_SD"


def register(profile):
    PROFILES[profile.key] = profile
    return profile


register(ImageProfile(
    key="Muestra_SD",
    folder="CS",
    label="Muestra_SD",
    description="Muestra de soldadura (probeta vista desde arriba): cordon oscuro sobre fondo claro, con zonas mas reflectantes.",
    segment=segment_dark,
    params=dict(work_width=800, approx_eps=1.5,
                fine_sigma=4,         # suavizado del log-gris (px de trabajo)
                bg_sigma=35,          # escala del fondo local: mayor que la anchura del cordon
                safe=41,              # margen (px) del marco/borde donde se busca la semilla
                seed_pct=15,          # percentil mas oscuro del paso-banda que forma la semilla
                excl=61,              # distancia (px) al cordon a partir de la cual un pixel es "fondo"
                erode=15,             # erosion del cordon para medir su nivel interior
                frac_strong=0.5,      # borde seguro: fraccion del contraste cordon/fondo
                frac_weak=0.35,       # borde flojo (histeresis); menor = contorno mas holgado
                band=51,              # banda (px) alrededor de la mascara actual donde se refina
                level_sigma=45,       # escala (px) de los niveles locales de cordon y de fondo
                min_contrast=0.02,    # contraste minimo cordon/fondo (log-gris) para fijar umbrales
                iters=6),             # iteraciones de re-estimacion de fondo
))

register(ImageProfile(
    key="Corte_SD",
    folder="CC",
    label="Corte_SD",
    description="Corte de la soldadura con colores reales: cordon identificado por su textura.",
    segment=segment_texture,
    params=dict(work_width=800, approx_eps=1.5,
                tex_win=9,            # ventana (px de trabajo) de la desviacion tipica local
                tex_sigma=7,          # suavizado gaussiano del mapa de textura
                dark_ratio=0.6,       # intensidad / fondo local < este valor -> cavidad oscura del cordon
                dark_band=61,         # banda (px de trabajo) alrededor del nucleo donde se anexan cavidades
                grain_sigma=5,        # suavizado del tensor de estructura (grano horizontal)
                grain_frac=0.3,       # umbral de grano: 0 = nivel de la soldadura, 1 = nivel del metal base
                grain_band=61,        # banda (px de trabajo) alrededor del nucleo donde se busca la interfaz
                grain_open=31),       # apertura (px) para quitar apendices finos hacia el metal base
))


def detect_profile(path):
    """Perfil segun el nombre de la carpeta de la imagen (CS*, CC*). None si no coincide."""
    folder = Path(path).resolve().parent.name.upper()
    for profile in PROFILES.values():
        if folder.startswith(profile.folder.upper()):
            return profile
    return None


def detect_contour(img, profile, steps=None):
    """Devuelve (contorno en coordenadas de la imagen original, area_px_original)."""
    p = profile.params
    h0, w0 = img.shape[:2]
    scale = p["work_width"] / w0
    small = cv2.resize(img, (p["work_width"], int(round(h0 * scale))), interpolation=cv2.INTER_AREA)

    mask = profile.segment(small, p, steps)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0.0

    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, p["approx_eps"], True)

    # Escalar contorno al tamano original
    contour_full = np.round(contour.astype(np.float64) / scale).astype(np.int32)
    area_full = cv2.contourArea(contour_full)
    return contour_full, area_full


def draw_result(img, contour, area_px, pixel_area_mm2=None):
    """Dibuja contorno y area. Si se da pixel_area_mm2 (area de un pixel, mm2) el area se muestra en mm2."""
    result = img.copy()
    thickness = max(2, img.shape[1] // 400)
    if contour is not None:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), thickness)
        if pixel_area_mm2 is not None:
            text = f"Area: {area_px * pixel_area_mm2:,.3f} mm2"
        else:
            text = f"Area: {area_px:,.0f} px2"
        font_scale = img.shape[1] / 1500
        cv2.putText(result, text, (30, int(60 * font_scale) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(result, text, (30, int(60 * font_scale) + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return result


def process(path, profile=None, save_steps=False, image_size_mm=None):
    """Procesa una imagen. Si profile es None se deduce de la carpeta (o se usa el perfil por defecto).

    image_size_mm: (ancho_mm, alto_mm) que abarca la imagen completa. Con la resolucion de la imagen
    se obtiene el area de un pixel (mm2) y el area se muestra en mm2.
    Devuelve (contorno, area_px, imagen_resultado, ruta_salida).
    """
    path = Path(path)
    if profile is None:
        profile = detect_profile(path)
        if profile is None:
            profile = PROFILES[DEFAULT_PROFILE]
            print(f"[?] {path.name}: tipo no reconocido por la carpeta, se usa '{profile.key}'")
    img = load_image(path)
    pixel_area_mm2 = scale.pixel_area_mm2(img.shape, image_size_mm) if image_size_mm is not None else None
    steps = {} if save_steps else None
    contour, area = detect_contour(img, profile, steps)

    out_dir = profile.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_contorno.jpg"
    result = draw_result(img, contour, area, pixel_area_mm2)
    cv2.imwrite(str(out), result)

    if save_steps:
        steps_dir = out_dir / "pasos"
        steps_dir.mkdir(exist_ok=True)
        for name, step in steps.items():
            cv2.imwrite(str(steps_dir / f"{path.stem}_{name}.png"), step)

    if contour is None:
        print(f"[!] {path.name} ({profile.key}): no se detecto contorno")
    else:
        mm2 = f" = {area * pixel_area_mm2:,.3f} mm2" if pixel_area_mm2 is not None else ""
        print(f"[ok] {path.name} ({profile.key}): area = {area:,.0f} px2{mm2} -> {out}")
    return contour, area, result, out


def main():
    parser = argparse.ArgumentParser(description="Demo de contorno de cordon en forma de L")
    parser.add_argument("images", nargs="*", help="Imagenes a procesar (por defecto: las del perfil, FOTOS/<PERFIL>/*.tif)")
    parser.add_argument("--profile", choices=["auto", *PROFILES], default="auto",
                        help="Tipo de foto. auto: segun la carpeta de cada imagen")
    parser.add_argument("--list-profiles", action="store_true", help="Mostrar perfiles disponibles y salir")
    parser.add_argument("--steps", action="store_true", help="Guardar imagenes de pasos intermedios")
    args = parser.parse_args()

    if args.list_profiles:
        for p in PROFILES.values():
            print(f"{p.key:4s} {p.label:16s} {p.description}\n     carpeta: {p.input_dir}")
        return 0

    profile = None if args.profile == "auto" else PROFILES[args.profile]
    if args.images:
        files = [Path(p) for p in args.images]
    else:
        folder = (profile or PROFILES[DEFAULT_PROFILE]).input_dir
        files = sorted(folder.glob("*.tif"))
        if not files:
            print(f"No se encontraron imagenes en {folder}")
            return 1
    for f in files:
        process(f, profile, save_steps=args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
