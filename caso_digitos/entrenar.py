vscode_lsp_terminal_prompt_tracker= {}
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from preprocesamiento import normalizar, ESCALA_MAX

SEMILLA = 42
keras.utils.set_random_seed(SEMILLA)

# 1. DATOS Y DIVISIÓN 60/20/20 (estratificada y con semilla fija)
datos = load_digits()
X = normalizar(datos.data)
y = datos.target
indices = np.arange(len(y))

idx_desarrollo, idx_prueba = train_test_split(
    indices, test_size=0.20, random_state=SEMILLA, stratify=y
)
idx_entrena, idx_valida = train_test_split(
    idx_desarrollo, test_size=0.25, random_state=SEMILLA, stratify=y[idx_desarrollo]
)

np.savez("indices.npz", entrena=idx_entrena, valida=idx_valida, prueba=idx_prueba)

X_entrena, y_entrena = X[idx_entrena], y[idx_entrena]
X_valida, y_valida = X[idx_valida], y[idx_valida]
X_prueba, y_prueba = X[idx_prueba], y[idx_prueba]

print("Entrenamiento:", len(idx_entrena))
print("Validación:", len(idx_valida))
print("Prueba:", len(idx_prueba))

# 2. CONFIGURACIÓN DE LOS DOS MODELOS
CONFIG = {
    "secuencial": {"capas": [64, 32], "dropout": 0.2, "l2": 1e-4, "lr": 0.001},
    "funcional": {"capas": [128, 64], "dropout": 0.3, "l2": 1e-4, "lr": 0.001},
    "batch_size": 32,
    "epocas_max": 200,
    "paciencia": 15,
    "semilla": SEMILLA,
}


def crear_secuencial(c):
    reg = keras.regularizers.l2(c["l2"])
    capas = [keras.Input(shape=(64,))]
    for n in c["capas"]:
        capas.append(layers.Dense(n, activation="relu", kernel_regularizer=reg))
        capas.append(layers.Dropout(c["dropout"]))
    capas.append(layers.Dense(10, activation="softmax"))
    return keras.Sequential(capas, name="secuencial")


def crear_funcional(c):
    reg = keras.regularizers.l2(c["l2"])
    entrada = keras.Input(shape=(64,), name="pixeles")
    x = entrada
    for n in c["capas"]:
        x = layers.Dense(n, activation="relu", kernel_regularizer=reg)(x)
        x = layers.Dropout(c["dropout"])(x)
    salida = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs=entrada, outputs=salida, name="funcional")


def medir(modelo, X_, y_):
    pred = np.argmax(modelo.predict(X_, verbose=0), axis=1)
    return accuracy_score(y_, pred), f1_score(y_, pred, average="macro")


# 3. ENTRENAR LOS DOS
modelos = {}
historiales = {}
resultados = {}

for nombre, crear in [("secuencial", crear_secuencial), ("funcional", crear_funcional)]:
    c = CONFIG[nombre]
    modelo = crear(c)
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=c["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    parada = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["paciencia"],
        restore_best_weights=True
    )
    print("\nENTRENANDO:", nombre)
    h = modelo.fit(
        X_entrena, y_entrena,
        validation_data=(X_valida, y_valida),
        epochs=CONFIG["epocas_max"],
        batch_size=CONFIG["batch_size"],
        callbacks=[parada],
        verbose=0
    )
    acc_v, f1_v = medir(modelo, X_valida, y_valida)
    acc_p, f1_p = medir(modelo, X_prueba, y_prueba)

    modelos[nombre] = modelo
    historiales[nombre] = h.history
    resultados[nombre] = {
        "parametros": int(modelo.count_params()),
        "epocas_efectivas": len(h.history["loss"]),
        "acc_valida": float(acc_v),
        "f1_valida": float(f1_v),
        "acc_prueba": float(acc_p),
        "f1_prueba": float(f1_p),
    }

# 4. COMPARACIÓN
print("\nCOMPARACIÓN DE MODELOS")
print("modelo      params  épocas  acc_val  f1_val  acc_prueba  f1_prueba")
for nombre, r in resultados.items():
    print(f"{nombre:<11} {r['parametros']:<7} {r['epocas_efectivas']:<7} "
          f"{r['acc_valida']:.4f}   {r['f1_valida']:.4f}  "
          f"{r['acc_prueba']:.4f}      {r['f1_prueba']:.4f}")

# La red final se elige SOLO por validación
final = max(resultados, key=lambda n: (resultados[n]["f1_valida"], resultados[n]["acc_valida"]))
print("\nModelo elegido por validación:", final)
modelo_final = modelos[final]

# 5. CURVAS DE PÉRDIDA
fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
for ax, nombre in zip(ejes, ["secuencial", "funcional"]):
    ax.plot(historiales[nombre]["loss"], label="Entrenamiento")
    ax.plot(historiales[nombre]["val_loss"], label="Validación")
    ax.set_title("Pérdida: " + nombre)
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
fig.tight_layout()
fig.savefig("curvas_perdida.png", dpi=150)

# 6. MATRIZ DE CONFUSIÓN DEL MODELO FINAL (PRUEBA)
probs_prueba = modelo_final.predict(X_prueba, verbose=0)
pred_prueba = np.argmax(probs_prueba, axis=1)

matriz_confusion = confusion_matrix(y_prueba, pred_prueba, labels=list(range(10)))
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(matriz_confusion, display_labels=list(range(10))).plot(
    ax=ax, cmap="Blues", colorbar=False
)
ax.set_title("Matriz de confusión (prueba): " + final)
ax.set_xlabel("Dígito predicho")
ax.set_ylabel("Dígito real")
fig.tight_layout()
fig.savefig("matriz_confusion.png", dpi=150)

# 7. ANÁLISIS DE ERRORES (hasta 3)
errores = np.where(pred_prueba != y_prueba)[0][:3]
print("\nERRORES EN PRUEBA (hasta 3):", len(errores))
if len(errores) > 0:
    fig, ejes = plt.subplots(1, len(errores), figsize=(3 * len(errores), 3.5), squeeze=False)
    for ax, i in zip(ejes[0], errores):
        ax.imshow(datos.images[idx_prueba[i]], cmap="gray_r")
        ax.set_title(f"Real {y_prueba[i]} | Pred {pred_prueba[i]}\n"
                     f"prob {probs_prueba[i].max():.2f}")
        ax.axis("off")
        print(f"Imagen {idx_prueba[i]}: real={y_prueba[i]}, "
              f"predicho={pred_prueba[i]}, prob={probs_prueba[i].max():.2f}")
    fig.tight_layout()
    fig.savefig("errores.png", dpi=150)

# 8. GUARDAR MODELO, PREPROCESAMIENTO, RESULTADOS Y 5 MUESTRAS FIJAS
modelo_final.save("modelo_digitos.keras")

with open("preproc.json", "w", encoding="utf-8") as f:
    json.dump({
        "entrada": 64,
        "forma_imagen": [8, 8],
        "escala_original": [0, ESCALA_MAX],
        "escala_modelo": [0, 1],
        "operacion": "pixel / 16",
        "modelo": final
    }, f, indent=2, ensure_ascii=False)

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump({"config": CONFIG, "resultados": resultados, "elegido": final},
              f, indent=2, ensure_ascii=False)

muestras = X_prueba[:5]
np.savez("muestras_fijas.npz", X=muestras, probs=modelo_final.predict(muestras, verbose=0))

print("\nListo: modelo_digitos.keras, preproc.json, indices.npz, resultados.json y gráficas guardadas.")
plt.show()