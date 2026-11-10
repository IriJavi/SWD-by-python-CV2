"""Escala fisica de una imagen: de pixeles a mm, a partir del tamano real (mm) que abarca la imagen completa."""


def mm_per_pixel(image_shape, image_size_mm):
    """Devuelve (mm por pixel en X, mm por pixel en Y).

    image_shape: img.shape de OpenCV (alto_px, ancho_px, ...). image_size_mm: (ancho_mm, alto_mm).
    """
    h_px, w_px = image_shape[:2]
    width_mm, height_mm = image_size_mm
    return width_mm / w_px, height_mm / h_px


def pixel_area_mm2(image_shape, image_size_mm):
    """Area de un pixel en mm2."""
    sx, sy = mm_per_pixel(image_shape, image_size_mm)
    return sx * sy
