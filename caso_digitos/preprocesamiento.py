import numpy as np
from PIL import Image

ESCALA_MAX = 16.0
UMBRAL_TINTA = 200       # píxeles más oscuros que esto cuentan como trazo
MIN_PIXELES_TINTA = 30   # con menos trazo que esto, el lienzo se considera vacío


def normalizar(X):
    # Pasa los píxeles de la escala 0-16 a la escala 0-1
    return np.asarray(X, dtype="float32") / ESCALA_MAX


def lienzo_vacio(imagen):
    arr = np.asarray(imagen.convert("L"))
    return int((arr < UMBRAL_TINTA).sum()) < MIN_PIXELES_TINTA


def lienzo_a_8x8(imagen):
    # Dibujo (trazo negro sobre fondo blanco) -> arreglo 8x8 en escala 0-16
    arr = np.asarray(imagen.convert("L"), dtype="float32")

    # Inversión de colores: load_digits tiene el trazo alto y el fondo en 0
    tinta = 255.0 - arr

    # Recorte al rectángulo que contiene el trazo
    ys, xs = np.where(arr < UMBRAL_TINTA)
    recorte = tinta[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Centrado: se pone en un cuadro negro con margen alrededor
    alto, ancho = recorte.shape
    lado = int(max(alto, ancho) * 1.25)
    cuadro = np.zeros((lado, lado), dtype="float32")
    oy = (lado - alto) // 2
    ox = (lado - ancho) // 2
    cuadro[oy:oy + alto, ox:ox + ancho] = recorte

    # Redimensionamiento a 8x8 promediando áreas
    img = Image.fromarray(cuadro.astype("uint8")).resize((8, 8), Image.Resampling.BOX)
    a = np.asarray(img, dtype="float32")

    # Se lleva a la escala 0-16 de load_digits
    if a.max() > 0:
        a = a / a.max() * ESCALA_MAX
    return a


def lienzo_a_vector(imagen):
    # Devuelve (vector de 64 valores en escala 0-1 con forma (1, 64), imagen 8x8)
    img8 = lienzo_a_8x8(imagen)
    return normalizar(img8.reshape(1, 64)), img8