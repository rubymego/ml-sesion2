import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow import keras

TOLERANCIA = 1e-5

datos = np.load("muestras_fijas.npz")
modelo = keras.models.load_model("modelo_digitos.keras")

probs_nuevas = modelo.predict(datos["X"], verbose=0)
diferencia = np.max(np.abs(probs_nuevas - datos["probs"]))
mismas_clases = np.array_equal(np.argmax(probs_nuevas, axis=1), np.argmax(datos["probs"], axis=1))

print("Tolerancia declarada:", TOLERANCIA)
print("Diferencia máxima:", diferencia)
print("Mismas clases predichas:", mismas_clases)
print("RESULTADO:", "conserva sus predicciones" if diferencia <= TOLERANCIA and mismas_clases else "NO coincide")