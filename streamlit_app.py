import streamlit as st
import clip
import cv2
import numpy as np
from PIL import Image

def generate_video(description, output_path, duration, fps):
    # Cargar el modelo CLIP
    model, preprocess = clip.load("ViT-B/32", device="cuda")

    # Generar una imagen inicial a partir de la descripción
    text = clip.tokenize([description]).to("cuda")
    with torch.no_grad():
        image = model.encode_text(text)
    image = image / image.norm(dim=-1, keepdim=True)

    # Configurar el video
    height, width = 512, 512
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Generar los cuadros del video
    num_frames = duration * fps
    for i in range(num_frames):
        # Generar una nueva imagen a partir de la descripción
        with torch.no_grad():
            noise = torch.randn_like(image) * 0.1
            new_image = model.decode_image(image + noise)
        new_image = new_image.clamp(0, 1)

        # Convertir la imagen a un formato compatible con OpenCV
        new_image = (new_image * 255).byte().cpu().numpy()
        new_image = np.transpose(new_image, (1, 2, 0))

        # Redimensionar la imagen al tamaño del video
        new_image = cv2.resize(new_image, (width, height))

        # Escribir el cuadro en el video
        video_writer.write(new_image)

        # Actualizar la imagen para el siguiente cuadro
        with torch.no_grad():
            image = model.encode_image(Image.fromarray(new_image).convert("RGB")).to("cuda")
        image = image / image.norm(dim=-1, keepdim=True)

    video_writer.release()

def main():
    st.title("Generador de videos a partir de descripciones")
    st.write("¡Bienvenido al Generador de videos a partir de descripciones!")
    st.write("Esta aplicación te permite generar un video a partir de una descripción utilizando el modelo CLIP de OpenAI.")

    # Obtener la descripción del usuario
    description = st.text_input("Descripción")

    # Configurar opciones del video
    duration = st.slider("Duración del video (segundos)", min_value=1, max_value=10, value=5)
    fps = st.slider("FPS (cuadros por segundo)", min_value=1, max_value=30, value=24)

    # Generar el video
    if st.button("Generar video"):
        output_path = "output.mp4"
        generate_video(description, output_path, duration, fps)
        st.success("¡Video generado con éxito!")
        st.download_button("Descargar video", output_path)

if __name__ == "__main__":
    main()
