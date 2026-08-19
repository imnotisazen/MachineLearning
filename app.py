import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Clasificador de Imágenes CIFAR-10",
    page_icon="🤖",
    layout="wide"
)

st.title(" Clasificador de Imágenes con IA")
st.markdown("### Desarrollado por: **Yolanda Martínez**")
st.markdown("---")

@st.cache_resource
def load_resources():
    """Carga el modelo y los nombres de clases"""
    try:
        # Verificar si existe el archivo del modelo
        if not os.path.exists('cifar10_model.h5'):
            st.error("❌ No se encuentra el archivo 'cifar10_model.h5'")
            st.info("Asegúrate de tener el modelo en el repositorio")
            return None, None
        
        # Cargar modelo
        model = load_model('cifar10_model.h5')
        
        # Cargar nombres de clases
        try:
            with open('class_names.pkl', 'rb') as f:
                class_names = pickle.load(f)
        except:
            # Si no existe el archivo, usar nombres por defecto
            class_names = ['Avión', 'Auto', 'Pájaro', 'Gato', 'Ciervo', 
                          'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']
        
        return model, class_names
    except Exception as e:
        st.error(f" Error al cargar el modelo: {e}")
        return None, None

model, class_names = load_resources()

def preprocess_image(image):
    """Prepara la imagen para el modelo"""
    try:
        image = image.resize((32, 32))
        image_array = np.array(image)
        image_array = image_array / 255.0
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array]*3, axis=-1)
        elif image_array.shape[-1] == 4:
            image_array = image_array[:,:,:3]
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        st.error(f" Error al procesar la imagen: {e}")
        return None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Subir imagen")
    
    # Subir archivo
    uploaded_file = st.file_uploader(
        "Elige una imagen...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    img = None
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption=' Imagen subida', use_container_width=True)
        except Exception as e:
            st.error(f" Error al abrir la imagen: {e}")
    
    st.markdown("---")
    st.markdown("###  El modelo puede clasificar:")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("✈️ Avión")
        st.markdown("🚗 Auto")
        st.markdown("🐦 Pájaro")
    with col_b:
        st.markdown("🐱 Gato")
        st.markdown("🦌 Ciervo")
        st.markdown("🐕 Perro")
    with col_c:
        st.markdown("🐸 Rana")
        st.markdown("🐴 Caballo")
        st.markdown("🚢 Barco")
        st.markdown("🚛 Camión")

with col2:
    st.subheader(" Predicción")
    
    # Botón para analizar
    if st.button("🔍 Analizar imagen", use_container_width=True, type="primary"):
        if img is not None:
            with st.spinner(" Analizando la imagen..."):
                # Preprocesar
                processed_img = preprocess_image(img)
                
                if processed_img is not None and model is not None:
                    try:
                        # Hacer predicción
                        prediction = model.predict(processed_img, verbose=0)
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        # Mostrar resultados
                        st.success("¡Imagen analizada!")
                        
                        # Resultado principal
                        st.markdown(f"###  **{class_names[predicted_class]}**")
                        st.markdown(f"###  Confianza: **{confidence:.2%}**")
                        
                        # Barra de progreso de la confianza
                        st.progress(float(confidence))
                        
                        # Mostrar todas las probabilidades
                        st.subheader("📊 Probabilidades por clase:")
                        
                        # Ordenar por probabilidad
                        sorted_indices = np.argsort(prediction[0])[::-1]
                        
                        for i in sorted_indices[:5]:  # Top 5
                            prob = prediction[0][i]
                            if prob > 0.01:  # Solo mostrar si > 1%
                                st.progress(float(prob), text=f"{class_names[i]}: {prob:.2%}")
                        
                        # Mostrar otras predicciones
                        if len(prediction[0]) > 5:
                            with st.expander(" Ver todas las clases"):
                                for i, (cls, prob) in enumerate(zip(class_names, prediction[0])):
                                    if prob > 0.01:
                                        st.progress(float(prob), text=f"{cls}: {prob:.2%}")
                    
                    except Exception as e:
                        st.error(f" Error en la predicción: {e}")
                else:
                    if model is None:
                        st.error(" El modelo no está cargado correctamente")
                    else:
                        st.warning(" No se pudo procesar la imagen")
        else:
            st.warning(" Por favor, sube una imagen primero")

st.markdown("---")
st.markdown("""
### ℹ️ Información del Sistema

| Característica | Descripción |
|---------------|-------------|
| **Modelo** | CNN entrenado con CIFAR-10 |
| **Clases** | 10 categorías de objetos |
| **Precisión** | ~70-75% en prueba |
| **Tamaño imagen** | 32x32 píxeles |
| **Framework** | TensorFlow + Streamlit |

###  Cómo usar:
1. Sube una imagen usando el botón
2. Haz clic en "Analizar imagen"
3. El modelo te dirá qué objeto es
4. La confianza muestra qué tan seguro está

**Consejo**: Para mejores resultados, usa imágenes claras y bien iluminadas.
""")

st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Desarrollado con ❤️ para el curso de Machine Learning</p>
    <p>© 2024 - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)
# Descargar app.py
files.download('app.py')
