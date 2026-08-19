# Crear el código para la aplicación Streamlit
app_code = '''
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Configurar la página
st.set_page_config(
    page_title="Clasificador de Imágenes CIFAR-10",
    page_icon="🤖",
    layout="wide"
)

# Título y descripción
st.title("🔍 Clasificador de Imágenes con IA")
st.markdown("### Desarrollado por: **Tu Nombre**")
st.markdown("---")

# Cargar el modelo y las clases
@st.cache_resource
def load_resources():
    try:
        model = load_model('cifar10_model.h5')
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        return model, class_names
    except:
        # Si no encuentra los archivos, usar los datos por defecto
        from tensorflow.keras.datasets import cifar10
        model = None
        class_names = ['Avión', 'Auto', 'Pájaro', 'Gato', 'Ciervo', 
                      'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']
        return model, class_names

model, class_names = load_resources()

# Función para preprocesar la imagen
def preprocess_image(image):
    # Redimensionar a 32x32
    image = image.resize((32, 32))
    # Convertir a array y normalizar
    image_array = np.array(image) / 255.0
    # Asegurar que tiene 3 canales
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array]*3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[:,:,:3]
    return np.expand_dims(image_array, axis=0)

# Layout de la aplicación
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Subir imagen")
    
    # Opciones para cargar imagen
    option = st.radio(
        "Selecciona el método de entrada:",
        ["Subir imagen desde archivo", "Usar imagen de ejemplo"]
    )
    
    img = None
    
    if option == "Subir imagen desde archivo":
        uploaded_file = st.file_uploader("Elige una imagen...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Imagen subida', use_container_width=True)
    else:
        # Mostrar imágenes de ejemplo
        example_images = [
            ("Ejemplo 1: Auto", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/2019-10-27_Red_Porsche_911_Carrera_4S.jpg/800px-2019-10-27_Red_Porsche_911_Carrera_4S.jpg"),
            ("Ejemplo 2: Pájaro", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/American_Robin_%28Turdus_migratorius%29_in_Spring_08.jpg/800px-American_Robin_%28Turdus_migratorius%29_in_Spring_08.jpg"),
            ("Ejemplo 3: Barco", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/SS_Great_Britain_2022.jpg/800px-SS_Great_Britain_2022.jpg")
        ]
        
        selected_example = st.selectbox("Selecciona una imagen de ejemplo:", 
                                       [img[0] for img in example_images])
        
        if selected_example:
            # Cargar imagen de ejemplo (en producción usarías URLs o imágenes locales)
            st.warning("En producción, estas imágenes de ejemplo estarían disponibles localmente")
            img = Image.open("ejemplo.jpg")  # Placeholder

with col2:
    st.subheader(" Predicción")
    
    if st.button(" Analizar imagen", use_container_width=True):
        if img is not None:
            with st.spinner("Analizando la imagen..."):
                # Preprocesar la imagen
                processed_img = preprocess_image(img)
                
                if model is not None:
                    # Hacer la predicción
                    prediction = model.predict(processed_img)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # Mostrar resultados
                    st.success(" Imagen analizada exitosamente!")
                    
                    # Mostrar la predicción principal
                    st.markdown(f"###  Resultado: **{class_names[predicted_class]}**")
                    st.markdown(f"###  Confianza: **{confidence:.2%}**")
                    
                    # Mostrar todas las probabilidades
                    st.subheader("Probabilidades por clase:")
                    
                    # Crear barras de probabilidad
                    for i, (cls, prob) in enumerate(zip(class_names, prediction[0])):
                        st.progress(float(prob), text=f"{cls}: {prob:.2%}")
                else:
                    st.error("El modelo no está cargado. Por favor, asegúrate de tener el archivo del modelo.")
        else:
            st.warning("Por favor, sube o selecciona una imagen primero.")

# Información adicional
st.markdown("---")
st.markdown("""
### Información
- **Modelo**: CNN entrenado con CIFAR-10 (10 clases)
- **Clases**: Avión, Auto, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión
- **Precisión**: ~70-75% en el conjunto de prueba
- **Tamaño de imagen**: 32x32 píxeles
""")
'''

# Guardar el archivo app.py
with open('app.py', 'w') as f:
    f.write(app_code)

# Descargar app.py
files.download('app.py')
