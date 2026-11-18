# pruebas.py - Pruebas con dibujos propios y ruido
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
import re
import numpy as np
from collections import Counter
from tensorflow import keras
from sklearn.datasets import load_digits

from preprocesamiento import normalizar

CARPETA = os.path.dirname(os.path.abspath(__file__))
SEMILLA = 42
UMBRAL = 0.70
SIGMA = 0.15
CAMPOS = ["fecha_hora", "etiqueta", "prediccion", "prob_max", "resultado"]

modelo = keras.models.load_model(os.path.join(CARPETA, "modelo_digitos.keras"))

# 1. DIBUJOS PROPIOS (primer intento de cada uno: últimas 20 filas del historial)
# Solo se leen las líneas que empiezan con una fecha, así se ignora el encabezado y líneas raras
patron = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},")
with open(os.path.join(CARPETA, "historial.csv"), encoding="utf-8-sig") as f:
    lineas = [l.strip() for l in f if patron.match(l.strip())]
filas = list(csv.DictReader(lineas, fieldnames=CAMPOS))[-20:]

print("DIBUJOS PROPIOS")
if len(filas) < 20:
    print("Aviso: hay solo", len(filas), "dibujos en el historial (se piden 20).")

por_digito = Counter(int(r["etiqueta"]) for r in filas)
print("Dibujos por dígito:", dict(sorted(por_digito.items())))

print("\nTabla de dibujos (dibujado | predicho | prob | acierto | baja confianza)")
correctos = 0
bajas = 0
for r in filas:
    ok = int(r["etiqueta"]) == int(r["prediccion"])
    baja = float(r["prob_max"]) < UMBRAL
    correctos += int(ok)
    bajas += int(baja)
    print(f"  {r['fecha_hora']} | {r['etiqueta']} | {r['prediccion']} | "
          f"{r['prob_max']} | {'si' if ok else 'NO'} | {'baja' if baja else ''}")

acc_dibujos = correctos / len(filas)
print(f"\nCorrectos: {correctos} de {len(filas)}")
print(f"Accuracy de dibujos propios: {acc_dibujos:.2%}")
print(f"Avisos de baja confianza (prob < {UMBRAL}): {bajas}")

# 2. RUIDO GAUSSIANO SOBRE 30 IMÁGENES DE PRUEBA (selección con semilla fija)
datos = load_digits()
indices = np.load(os.path.join(CARPETA, "indices.npz"))
idx_prueba = indices["prueba"]

rng = np.random.default_rng(SEMILLA)
elegidos = rng.choice(idx_prueba, size=30, replace=False)

X = normalizar(datos.data[elegidos])
y = datos.target[elegidos]

X_ruido = np.clip(X + rng.normal(0, SIGMA, X.shape), 0, 1).astype("float32")

pred_orig = np.argmax(modelo.predict(X, verbose=0), axis=1)
pred_ruido = np.argmax(modelo.predict(X_ruido, verbose=0), axis=1)

acc_orig = float(np.mean(pred_orig == y))
acc_ruido = float(np.mean(pred_ruido == y))
cambiaron = int(np.sum(pred_orig != pred_ruido))

print("\nRUIDO GAUSSIANO (sigma = 0.15, 30 imágenes)")
print(f"Accuracy originales: {acc_orig:.2%}")
print(f"Accuracy con ruido: {acc_ruido:.2%}")
print("Predicciones que cambiaron por el ruido:", cambiaron)
for i in np.where(pred_ruido != y)[0]:
    print(f"  Imagen {elegidos[i]}: real={y[i]}, original={pred_orig[i]}, con ruido={pred_ruido[i]}")

# 3. COMPARACIÓN CON LA PRUEBA COMPLETA
with open(os.path.join(CARPETA, "resultados.json"), encoding="utf-8") as f:
    res = json.load(f)
elegido = res["elegido"]
acc_prueba = res["resultados"][elegido]["acc_prueba"]

print("\nCOMPARACIÓN")
print(f"Prueba completa ({elegido}): {acc_prueba:.2%}")
print(f"Dibujos propios: {acc_dibujos:.2%}")
print(f"Imágenes con ruido: {acc_ruido:.2%}")

with open(os.path.join(CARPETA, "resultados_pruebas.json"), "w", encoding="utf-8") as f:
    json.dump({
        "dibujos_propios": {"total": len(filas), "correctos": correctos,
                            "accuracy": acc_dibujos, "avisos_baja_confianza": bajas},
        "ruido": {"sigma": SIGMA, "imagenes": 30, "accuracy_original": acc_orig,
                  "accuracy_con_ruido": acc_ruido},
        "prueba_completa": acc_prueba
    }, f, indent=2, ensure_ascii=False)