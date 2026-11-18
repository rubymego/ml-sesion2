vscode_lsp_terminal_prompt_tracker= {}
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

print("Librerías cargadas correctamente")
print("TensorFlow:", tf.__version__)

# PASO 2: CREAR UN VECTOR
# Medidas de una flor
vector = np.array([5.1, 3.5, 1.4, 0.2])

print("\nVECTOR:")
print(vector)
print("Forma del vector:", vector.shape)

# PASO 3: CREAR UNA MATRIZ
# Cada fila representa una flor
# Cada columna representa una característica
matriz = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3],
    [6.3, 3.3, 6.0, 2.5]
])

print("\nMATRIZ:")
print(matriz)
print("Forma de la matriz:", matriz.shape)

# PASO 5: CREAR PESOS Y SESGOS
# Cuatro características de entrada y tres salidas
pesos = np.array([
    [0.2, 0.4, 0.1],
    [0.3, 0.1, 0.5],
    [0.4, 0.2, 0.3],
    [0.1, 0.3, 0.2]
])

sesgo = np.array([0.1, 0.2, 0.3])

print("\nPESOS:")
print(pesos)
print("Forma de los pesos:", pesos.shape)

# PASO 6: MULTIPLICAR Y SUMAR EL SESGO
salida = matriz @ pesos + sesgo

print("\nSALIDA:")
print(salida)
print("Forma de la salida:", salida.shape)

# PASO 7: CARGAR LOS DATOS IRIS
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()

X = iris.data.astype("float32")
y = iris.target

print("\nDATOS IRIS")
print("Forma de las entradas:", X.shape)
print("Forma de las etiquetas:", y.shape)
print("Especies:", iris.target_names)
print("Primeras cinco flores:\n", X[:5])

# PASO 8: SEPARAR LOS DATOS
# Reservamos 20 % para prueba
X_desarrollo, X_prueba, y_desarrollo, y_prueba = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Del resto, reservamos 25 % para validación
X_entrena, X_valida, y_entrena, y_valida = train_test_split(
    X_desarrollo,
    y_desarrollo,
    test_size=0.25,
    random_state=42,
    stratify=y_desarrollo
)

print("\nDISTRIBUCIÓN DE DATOS")
print("Entrenamiento:", len(X_entrena))
print("Validación:", len(X_valida))
print("Prueba:", len(X_prueba))

# PASO 9: ESCALAR LAS CARACTERÍSTICAS
# Aprendemos la transformación solo con entrenamiento
escalador = StandardScaler()

X_entrena = escalador.fit_transform(X_entrena)
X_valida = escalador.transform(X_valida)
X_prueba = escalador.transform(X_prueba)

print("\nPrimera flor de entrenamiento escalada:")
print(X_entrena[0])

# PASO 10: CREAR LA RED NEURONAL
modelo = keras.Sequential([
    keras.Input(shape=(4,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(3, activation="softmax")
])

print("\nRESUMEN DE LA RED")
modelo.summary()

# PASO 11: CONFIGURAR EL APRENDIZAJE
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# PASO 12: ENTRENAR
parada = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

print("\nINICIANDO ENTRENAMIENTO")

historial = modelo.fit(
    X_entrena,
    y_entrena,
    validation_data=(X_valida, y_valida),
    epochs=200,
    batch_size=16,
    callbacks=[parada],
    verbose=1
)

# PASO 13: EVALUAR CON LOS DATOS DE PRUEBA
perdida, exactitud = modelo.evaluate(
    X_prueba,
    y_prueba,
    verbose=0
)

print(f"\nExactitud en prueba: {exactitud:.2%}")

if exactitud >= 0.85:
    print("Se alcanzó la meta del 85 %")
else:
    print("No se alcanzó la meta del 85 %")

# PASO 14: OBTENER LAS PREDICCIONES
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

probabilidades = modelo.predict(X_prueba, verbose=0)
predicciones = np.argmax(probabilidades, axis=1)

print("\nPRIMERAS 5 PREDICCIONES")
for i in range(5):
    real = iris.target_names[y_prueba[i]]
    predicha = iris.target_names[predicciones[i]]
    print(f"Real: {real} | Predicción: {predicha}")

# PASO 15: MOSTRAR LA MATRIZ DE CONFUSIÓN
matriz_confusion = confusion_matrix(
    y_prueba,
    predicciones,
    labels=[0, 1, 2]
)

print("\nMATRIZ DE CONFUSIÓN")
print(matriz_confusion)

fig, ax = plt.subplots(figsize=(7, 5))
visualizador = ConfusionMatrixDisplay(
    confusion_matrix=matriz_confusion,
    display_labels=iris.target_names
)
visualizador.plot(ax=ax, cmap="Blues", colorbar=False)

ax.set_title("Clasificación de flores Iris")
ax.set_xlabel("Especie predicha")
ax.set_ylabel("Especie real")

fig.tight_layout()
fig.savefig("matriz_confusion.png", dpi=150)

# PASO 16: GRAFICAR EL APRENDIZAJE
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(historial.history["loss"], label="Entrenamiento")
ax.plot(historial.history["val_loss"], label="Validación")

ax.set_title("Pérdida durante el aprendizaje")
ax.set_xlabel("Época")
ax.set_ylabel("Pérdida")
ax.legend()

fig.tight_layout()
fig.savefig("curva_aprendizaje.png", dpi=150)

plt.show()

# PASO 17: CREAR EL MODELO FUNCIONAL
entrada = keras.Input(shape=(4,), name="medidas_flor")

capa_oculta = layers.Dense(
    16,
    activation="relu"
)(entrada)

salida_funcional = layers.Dense(
    3,
    activation="softmax"
)(capa_oculta)

modelo_funcional = keras.Model(
    inputs=entrada,
    outputs=salida_funcional,
    name="clasificador_funcional"
)

print("\nRESUMEN DEL MODELO FUNCIONAL")
modelo_funcional.summary()

# PASO 18: CONFIGURAR EL MODELO
modelo_funcional.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Un callback nuevo para este entrenamiento
parada_funcional = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# PASO 19: ENTRENAR
historial_funcional = modelo_funcional.fit(
    X_entrena,
    y_entrena,
    validation_data=(X_valida, y_valida),
    epochs=200,
    batch_size=16,
    callbacks=[parada_funcional],
    verbose=1
)

# PASO 20: EVALUAR Y COMPARAR
perdida_funcional, exactitud_funcional = modelo_funcional.evaluate(
    X_prueba,
    y_prueba,
    verbose=0
)

print("\nCOMPARACIÓN DE MODELOS")
print(f"Secuencial: {exactitud:.2%}")
print(f"Funcional: {exactitud_funcional:.2%}")

print("Parámetros del secuencial:", modelo.count_params())
print("Parámetros del funcional:", modelo_funcional.count_params())

print("Épocas del secuencial:", len(historial.history["loss"]))
print("Épocas del funcional:", len(historial_funcional.history["loss"]))

# PASO 21: CREAR DATOS SIMULADOS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
generador = np.random.default_rng(42)
horas = np.arange(1, 41, dtype=float).reshape(-1, 1)
ruido = generador.normal(0, 4, 40)
consumo = 3.5 * horas.ravel() + 12 + ruido

# PASO 22: SEPARAR ENTRENAMIENTO Y PRUEBA
horas_entrena, horas_prueba, consumo_entrena, consumo_prueba = train_test_split(
    horas,
    consumo,
    test_size=0.20,
    random_state=42
)

# PASO 23: ENTRENAR LA REGRESIÓN
regresion = LinearRegression()
regresion.fit(horas_entrena, consumo_entrena)

print("\nREGRESIÓN LINEAL")
print("Pendiente:", regresion.coef_[0])
print("Intercepto:", regresion.intercept_)

# PASO 24: PREDECIR Y EVALUAR
consumo_predicho = regresion.predict(horas_prueba)

mae = mean_absolute_error(consumo_prueba, consumo_predicho)
r2 = r2_score(consumo_prueba, consumo_predicho)

print(f"Error absoluto medio: {mae:.3f} kWh")
print(f"R²: {r2:.3f}")

prediccion_20 = regresion.predict([[20]])[0]
print(f"Consumo estimado para 20 horas: {prediccion_20:.2f} kWh")

# PASO 25: GRAFICAR LOS RESULTADOS
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    horas_entrena,
    consumo_entrena,
    label="Entrenamiento",
    color="blue"
)

ax.scatter(
    horas_prueba,
    consumo_prueba,
    label="Prueba",
    color="orange"
)

ax.plot(
    horas,
    regresion.predict(horas),
    label="Recta ajustada",
    color="red"
)

ax.set_title("Regresión lineal del consumo de energía")
ax.set_xlabel("Horas de uso")
ax.set_ylabel("Consumo de energía (kWh)")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("regresion_lineal.png", dpi=150)
plt.show()