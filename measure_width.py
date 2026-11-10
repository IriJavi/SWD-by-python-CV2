"""
Medicion de la anchura del cordon (forma de "J") en tres puntos: parte superior, curva y punta inferior.

Modulo independiente de demo_contour.py (que solo calcula el area); solo reutiliza su deteccion de contorno.

Metodo (analisis de la FORMA, no del contorno suelto):
  1. Mascara rellena del cordon (contorno de demo_contour) y suavizado ligero (cierre) para quitar muescas.
  2. Eje central (esqueleto) y recorrido mas largo entre sus dos extremos -> linea media ordenada del cordon.
  3. Tres estaciones sobre la linea media: parte superior, punto de maxima curvatura (la curva) y punta.
  4. En cada estacion se traza la NORMAL a la linea media a ambos lados hasta salir de la mascara; la distancia
     entre los dos puntos de salida es la anchura (equivale a la mediatriz de la propuesta inicial, sin depender
     de detectar esquinas). Se promedia en una ventana corta para no depender de un rizado local del borde.
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import skeletonize

import demo_contour as dc
import scale

PARAMS = dict(
    skel_width=800,        # ancho (px) al que se calcula el esqueleto
    smooth_frac=0.012,     # cierre de la mascara (fraccion del ancho de la imagen) contra muescas del borde
    path_sigma=0.15,       # suavizado de la linea media para las normales (x anchura tipica)
    curve_sigma=0.40,      # suavizado de la linea media para localizar la curva (x anchura tipica)
    end_margin=1.0,        # distancia de la estacion superior al extremo de la linea media (x anchura)
    tip_margin=1.7,        # idem para la punta: mas hacia el centro para evitar poros/forma del extremo
    avg_window=0.25,       # semiventana de promediado alrededor de cada estacion (x anchura)
    avg_samples=7,         # numero de normales promediadas por estacion
    bar_near=1.0,          # orientacion de los extremos: tramo recto de la barra entre bar_near y bar_far
    bar_far=2.5,           #   (x anchura) hacia dentro desde la estacion
    end_tilt=20.0,         # en los extremos se busca la cuerda mas corta hasta +-end_tilt grados de la normal
)

STATIONS = ("superior", "curva", "punta")


# ---------------------------------------------------------------- geometria
def _farthest(start, neighbors):
    """BFS desde start. Devuelve (nodo mas lejano, padres)."""
    parent = {start: None}
    queue = deque([start])
    last = start
    while queue:
        node = queue.popleft()
        last = node
        for nb in neighbors[node]:
            if nb not in parent:
                parent[nb] = node
                queue.append(nb)
    return last, parent


def _longest_path(skel):
    """Camino mas largo (extremo a extremo) del esqueleto -> array (N, 2) de (x, y). Descarta ramas cortas."""
    pts = {tuple(p) for p in np.argwhere(skel)}
    neighbors = {p: [(p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if (dy or dx) and (p[0] + dy, p[1] + dx) in pts] for p in pts}
    a, _ = _farthest(next(iter(pts)), neighbors)
    b, parent = _farthest(a, neighbors)
    path = []
    while b is not None:
        path.append(b)
        b = parent[b]
    return np.array([(x, y) for y, x in path], dtype=np.float64)


def _resample(path, step=1.0):
    """Remuestrea la polilinea a paso constante de longitud de arco."""
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def _smooth(path, sigma):
    return np.column_stack([gaussian_filter1d(path[:, i], sigma, mode="nearest") for i in range(2)])


def _tangents(path):
    t = np.gradient(path, axis=0)
    return t / np.maximum(np.hypot(t[:, 0], t[:, 1])[:, None], 1e-9)


def _curvature(path):
    dx, dy = np.gradient(path[:, 0]), np.gradient(path[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return np.abs(dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)


def _ray_exit(mask, p, d, step=0.5):
    """Avanza desde p en la direccion d hasta salir de la mascara. Devuelve el ultimo punto dentro."""
    h, w = mask.shape
    q = p.copy()
    while True:
        nxt = q + d * step
        x, y = int(round(nxt[0])), int(round(nxt[1]))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            return q
        q = nxt


def _cross_section(mask, c, tangent, max_tilt=0.0):
    """Segmento perpendicular a la linea media en c: (extremo -, extremo +).

    Con max_tilt > 0 (grados) se prueban tambien cuerdas giradas hasta max_tilt respecto a la normal y se
    devuelve la mas corta: en una barra recta la cuerda mas corta es la perpendicular, lo que compensa que la
    linea media se curve cerca de los extremos.
    """
    n = np.array([-tangent[1], tangent[0]])
    if max_tilt <= 0:
        return _ray_exit(mask, c, -n), _ray_exit(mask, c, n)
    best = None
    for deg in np.arange(-max_tilt, max_tilt + 0.1, 2.0):
        r = np.deg2rad(deg)
        d = np.array([n[0] * np.cos(r) - n[1] * np.sin(r), n[0] * np.sin(r) + n[1] * np.cos(r)])
        a, b = _ray_exit(mask, c, -d, 1.0), _ray_exit(mask, c, d, 1.0)
        length = np.hypot(*(b - a))
        if best is None or length < best[0]:
            best = (length, d)
    d = best[1]
    return _ray_exit(mask, c, -d), _ray_exit(mask, c, d)


# ------------------------------------------------------------------ medicion
def _centerline(mask_full, p):
    """Linea media ordenada (de arriba a abajo, coordenadas de imagen completa) y anchura tipica (px)."""
    h0, w0 = mask_full.shape
    k = p["skel_width"] / w0
    small = cv2.resize(mask_full, (p["skel_width"], int(round(h0 * k))), interpolation=cv2.INTER_AREA) > 127
    path = _longest_path(skeletonize(small))
    if path[0, 1] > path[-1, 1]:      # orientar: el inicio es el extremo superior (menor y)
        path = path[::-1]
    dist = cv2.distanceTransform(small.astype(np.uint8), cv2.DIST_L2, 5)
    width_typ = 2 * np.median(dist[path[:, 1].astype(int), path[:, 0].astype(int)])
    return path / k, width_typ / k


def measure(img, contour, image_size_mm, params=None):
    """Mide la anchura en las tres estaciones.

    Devuelve (widths_mm, geometria) con geometria = {"centerline": (N,2), "segments": {nombre: (a, b)}}.
    """
    p = dict(PARAMS, **(params or {}))
    h0, w0 = img.shape[:2]

    mask = np.zeros((h0, w0), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
    ksz = max(3, int(p["smooth_frac"] * w0) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    mask_bool = mask > 127

    center, width_typ = _centerline(mask, p)
    center = _resample(center, 1.0)                    # paso de 1 px de imagen completa
    path = _smooth(center, p["path_sigma"] * width_typ)
    curve_path = _smooth(center, p["curve_sigma"] * width_typ)
    tangents = _tangents(path)

    n = len(path)
    margin = int(p["end_margin"] * width_typ)
    kappa = _curvature(curve_path)
    lo, hi = int(width_typ), n - int(width_typ)        # evitar efectos de extremo al buscar la curva
    i_curve = lo + int(np.argmax(kappa[lo:hi]))
    stations = {"superior": margin, "curva": i_curve, "punta": n - 1 - int(p["tip_margin"] * width_typ)}

    sx, sy = scale.mm_per_pixel(img.shape, image_size_mm)
    half = int(p["avg_window"] * width_typ)
    near, far = int(p["bar_near"] * width_typ), int(p["bar_far"] * width_typ)

    def bar_tangent(i_from, i_to):
        d = path[min(i_to, n - 1)] - path[min(max(i_from, 0), n - 1)]
        return d / np.hypot(*d)

    # Los extremos (tapa redondeada, poros) desvian el esqueleto: la orientacion de la estacion superior se toma
    # de la parte recta de la barra, mas hacia dentro. En la punta (barra corta, la curva queda cerca) y en la
    # curva se usa la tangente local.
    fixed = {"superior": bar_tangent(stations["superior"] + near, stations["superior"] + far)}
    tilt = {"superior": p["end_tilt"], "punta": p["end_tilt"]}

    widths, segments = {}, {}
    for name, i0 in stations.items():
        values = []
        for i in np.linspace(max(i0 - half, 0), min(i0 + half, n - 1), p["avg_samples"]).round().astype(int):
            a, b = _cross_section(mask_bool, path[i], fixed.get(name, tangents[i]), tilt.get(name, 0.0))
            d = b - a
            values.append(np.hypot(d[0] * sx, d[1] * sy))
        widths[name] = float(np.mean(values))
        segments[name] = _cross_section(mask_bool, path[i0], fixed.get(name, tangents[i0]), tilt.get(name, 0.0))

    return widths, {"centerline": path, "segments": segments}


# ------------------------------------------------------------------- dibujo
def draw_measurements(img, contour, widths, geometry):
    result = img.copy()
    t = max(2, img.shape[1] // 600)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 1800

    cv2.drawContours(result, [contour], -1, (0, 255, 0), t)
    cv2.polylines(result, [np.round(geometry["centerline"]).astype(np.int32).reshape(-1, 1, 2)],
                  False, (255, 128, 0), t)

    def label(text, org, color):
        cv2.putText(result, text, org, font, fs, (0, 0, 0), t + 3, cv2.LINE_AA)
        cv2.putText(result, text, org, font, fs, color, t, cv2.LINE_AA)

    for k, name in enumerate(STATIONS, 1):
        a, b = geometry["segments"][name]
        pa, pb = tuple(int(v) for v in np.round(a)), tuple(int(v) for v in np.round(b))
        cv2.line(result, pa, pb, (0, 0, 255), t + 1, cv2.LINE_AA)
        cv2.circle(result, pa, t + 3, (0, 255, 255), -1)
        cv2.circle(result, pb, t + 3, (0, 255, 255), -1)
        label(str(k), (pb[0] + 20, pb[1]), (0, 255, 255))
        label(f"{k}. {name}: {widths[name]:.3f} mm", (30, int(60 * fs) * k + 20), (0, 0, 255))
    return result


# --------------------------------------------------------------- operacion
def measure_width(path, profile, image_size_mm):
    """Mide la anchura del cordon en tres zonas.

    path: ruta de la imagen. profile: ImageProfile (o None = auto). image_size_mm: (ancho_mm, alto_mm) de la
    imagen completa.

    Devuelve (widths_mm, result_path):
        widths_mm   = {"superior": float, "curva": float, "punta": float}
        result_path = ruta de la imagen resultado con las mediciones dibujadas
    """
    path = Path(path)
    if profile is None:
        profile = dc.detect_profile(path) or dc.PROFILES[dc.DEFAULT_PROFILE]
    img = dc.load_image(path)
    contour, _ = dc.detect_contour(img, profile)
    if contour is None:
        raise ValueError("no se detecto contorno")

    widths, geometry = measure(img, contour, image_size_mm)

    out_dir = profile.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_anchura.jpg"
    cv2.imwrite(str(out), draw_measurements(img, contour, widths, geometry))
    print(f"[ok] {path.name} ({profile.key}): " + ", ".join(f"{k} = {v:.3f} mm" for k, v in widths.items())
          + f" -> {out}")
    return widths, out
