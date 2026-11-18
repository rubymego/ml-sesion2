# app.py - Reconocimiento de dígitos
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from preprocesamiento import lienzo_vacio, lienzo_a_vector

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA, "modelo_digitos.keras")
RUTA_HISTORIAL = os.path.join(CARPETA, "historial.csv")
UMBRAL_CONFIANZA = 0.70
TAM = 280
GROSOR = 20


class App:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("Reconocimiento de dígitos")

        self.modelo = self.cargar_modelo()
        self.imagen = Image.new("L", (TAM, TAM), 255)
        self.dibujo = ImageDraw.Draw(self.imagen)
        self.ultimo = None
        self.x_ant = None
        self.y_ant = None

        self.construir()

    def cargar_modelo(self):
        if not os.path.exists(RUTA_MODELO):
            messagebox.showerror(
                "Modelo ausente",
                "No se encontró modelo_digitos.keras.\nEjecuta primero: python entrenar.py"
            )
            return None
        try:
            from tensorflow import keras
            return keras.models.load_model(RUTA_MODELO)
        except Exception as e:
            messagebox.showerror("Error al cargar el modelo", str(e))
            return None

    def construir(self):
        izq = tk.Frame(self.raiz, padx=10, pady=10)
        izq.grid(row=0, column=0, sticky="n")
        der = tk.Frame(self.raiz, padx=10, pady=10)
        der.grid(row=0, column=1, sticky="n")

        tk.Label(izq, text="Dibuja un dígito del 0 al 9", font=("Arial", 12)).pack()
        self.lienzo = tk.Canvas(izq, width=TAM, height=TAM, bg="white",
                                highlightthickness=1, highlightbackground="black")
        self.lienzo.pack(pady=5)
        self.lienzo.bind("<B1-Motion>", self.dibujar)
        self.lienzo.bind("<ButtonRelease-1>", self.soltar)

        botones = tk.Frame(izq)
        botones.pack(pady=5)
        tk.Button(botones, text="Limpiar", width=12, command=self.limpiar).grid(row=0, column=0, padx=5)
        tk.Button(botones, text="Reconocer", width=12, command=self.reconocer).grid(row=0, column=1, padx=5)

        tk.Label(izq, text="Dígito que quise dibujar:").pack(pady=(15, 2))
        self.combo = ttk.Combobox(izq, values=[str(i) for i in range(10)],
                                  state="readonly", width=5)
        self.combo.pack()
        tk.Button(izq, text="Guardar en historial", command=self.guardar).pack(pady=8)

        self.lbl_pred = tk.Label(der, text="Predicción: -", font=("Arial", 22, "bold"))
        self.lbl_pred.pack()
        self.lbl_top3 = tk.Label(der, text="", font=("Arial", 11), justify="left")
        self.lbl_top3.pack()
        self.lbl_aviso = tk.Label(der, text="", font=("Arial", 11, "bold"), fg="red")
        self.lbl_aviso.pack()

        self.fig = Figure(figsize=(6, 3))
        self.ax_img = self.fig.add_subplot(1, 2, 1)
        self.ax_bar = self.fig.add_subplot(1, 2, 2)
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=der)
        self.canvas_fig.get_tk_widget().pack()
        self.dibujar_vacio()

    def dibujar_vacio(self):
        self.ax_img.clear()
        self.ax_bar.clear()
        self.ax_img.set_title("Imagen 8x8 del modelo", fontsize=9)
        self.ax_img.axis("off")
        self.ax_bar.set_title("Probabilidades", fontsize=9)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_xticks(range(10))
        self.fig.tight_layout()
        self.canvas_fig.draw()

    def dibujar(self, e):
        r = GROSOR / 2
        if self.x_ant is not None:
            self.lienzo.create_line(self.x_ant, self.y_ant, e.x, e.y, width=GROSOR, fill="black")
            self.dibujo.line([self.x_ant, self.y_ant, e.x, e.y], fill=0, width=GROSOR)
        self.lienzo.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="black", outline="black")
        self.dibujo.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=0)
        self.x_ant, self.y_ant = e.x, e.y

    def soltar(self, e):
        self.x_ant = None
        self.y_ant = None

    def limpiar(self):
        self.lienzo.delete("all")
        self.imagen = Image.new("L", (TAM, TAM), 255)
        self.dibujo = ImageDraw.Draw(self.imagen)
        self.ultimo = None
        self.lbl_pred.config(text="Predicción: -")
        self.lbl_top3.config(text="")
        self.lbl_aviso.config(text="")
        self.dibujar_vacio()

    def reconocer(self):
        if self.modelo is None:
            messagebox.showerror("Modelo ausente",
                                 "No hay modelo cargado. Ejecuta primero: python entrenar.py")
            return
        if lienzo_vacio(self.imagen):
            messagebox.showwarning("Lienzo vacío", "Dibuja un dígito antes de reconocer.")
            return

        x, img8 = lienzo_a_vector(self.imagen)
        probs = self.modelo.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        pmax = float(probs[pred])
        self.ultimo = (pred, pmax)

        top3 = np.argsort(probs)[::-1][:3]
        self.lbl_pred.config(text=f"Predicción: {pred}")
        self.lbl_top3.config(text="\n".join(
            f"{i + 1}. Dígito {d}: {probs[d]:.1%}" for i, d in enumerate(top3)))
        if pmax < UMBRAL_CONFIANZA:
            self.lbl_aviso.config(text=f"Aviso: baja confianza (menor a {UMBRAL_CONFIANZA:.2f})")
        else:
            self.lbl_aviso.config(text="")

        self.ax_img.clear()
        self.ax_img.imshow(img8, cmap="gray_r", vmin=0, vmax=16)
        self.ax_img.set_title("Imagen 8x8 del modelo", fontsize=9)
        self.ax_img.axis("off")

        self.ax_bar.clear()
        self.ax_bar.bar(range(10), probs, color="steelblue")
        self.ax_bar.axhline(UMBRAL_CONFIANZA, color="red", linestyle="--", linewidth=0.8)
        self.ax_bar.set_ylim(0, 1)
        self.ax_bar.set_xticks(range(10))
        self.ax_bar.set_title("Probabilidades", fontsize=9)
        self.fig.tight_layout()
        self.canvas_fig.draw()

    def guardar(self):
        if self.ultimo is None:
            messagebox.showinfo("Sin predicción", "Primero reconoce un dígito.")
            return
        texto = self.combo.get()
        if not texto.isdigit() or not (0 <= int(texto) <= 9):
            messagebox.showerror("Etiqueta no válida", "Elige un dígito del 0 al 9.")
            return

        etiqueta = int(texto)
        pred, pmax = self.ultimo
        resultado = "correcto" if etiqueta == pred else "incorrecto"
        nuevo = not os.path.exists(RUTA_HISTORIAL)
        try:
            with open(RUTA_HISTORIAL, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if nuevo:
                    w.writerow(["fecha_hora", "etiqueta", "prediccion", "prob_max", "resultado"])
                w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            etiqueta, pred, f"{pmax:.4f}", resultado])
        except OSError as e:
            messagebox.showerror("No se pudo guardar",
                                 "Cierra historial.csv si lo tienes abierto.\n" + str(e))
            return

        self.ultimo = None
        messagebox.showinfo("Guardado", f"Registrado: dibujado {etiqueta}, predicho {pred} ({resultado}).")


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()