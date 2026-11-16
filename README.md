# 🏛️ Zenon

**Chatbot de filosofía estoica basado en un modelo GPT-2 ajustado con citas de Marco Aurelio, Séneca y Epicteto.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-GPT--2-yellow)

Zenon, llamado así por Zenón de Citio, fundador del estoicismo, ajusta GPT-2 con un corpus de citas estoicas y ofrece una interfaz de escritorio para conversar con el modelo.

<p align="center">
  <img src="assets/avatar.jpg" alt="Avatar de Zenon" width="220">
</p>

## ✨ Características

- *Fine-tuning* de GPT-2 con PyTorch, *padding* dinámico y planificador lineal con calentamiento.
- Generación con muestreo *nucleus* (top-p) y bloqueo de n-gramas repetidos para respuestas más naturales.
- Interfaz gráfica con Tkinter: escribe tu pregunta y pulsa **Enter** o **Ask**.
- El modelo se carga una sola vez al iniciar el chat.

## 🚀 Instalación

```bash
git clone https://github.com/NicolasAyalaTovar/Zenon.git
cd Zenon
pip install -r requirements.txt
```

En Linux, si falta Tkinter: `sudo apt install python3-tk`.

## ▶️ Uso

**1. Entrenar el modelo.** Genera la carpeta `fitted_model/`:

```bash
python zenon.py train            # opcional: --epochs 4 --batch-size 2 --lr 5e-5
```

**2. Abrir el chat:**

```bash
python zenon.py chat
```

Para ampliar el corpus, añade archivos `.txt` a `data/` con una cita por línea y vuelve a entrenar.

## 📁 Estructura

```
Zenon/
├── zenon.py          # Entrenamiento (train) e interfaz de chat (chat)
├── data/             # Corpus de citas estoicas (.txt, una por línea)
├── assets/
│   └── avatar.jpg    # Imagen de la interfaz
└── requirements.txt
```

## ⚠️ Limitaciones

GPT-2 es un modelo pequeño y el corpus tiene unas pocas centenas de citas. Las respuestas imitan el estilo estoico, pero no deben tomarse como fuente filosófica fiable.

## 🔭 Mejoras futuras

- Ampliar el corpus con obras completas (*Meditaciones*, *Cartas a Lucilio*, *Enquiridión*).
- Recuperación de citas relevantes (RAG) para respuestas más fundamentadas.
- Interfaz web con Gradio o Streamlit.

## 👤 Autor

**Nicolás Ayala Tovar** · [GitHub](https://github.com/NicolasAyalaTovar)
