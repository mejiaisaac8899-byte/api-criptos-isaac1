import os
import io
import shutil
import glob
import numpy as np
import cv2
import gradio as gr
from fastapi import FastAPI, Request, Response
import onnxruntime as ort
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ==================================================
DATABASE_DIR = "database"
if not os.path.exists(DATABASE_DIR):
    os.makedirs(DATABASE_DIR)

SIMILARITY_THRESHOLD = 0.93

# Memoria temporal para la foto del ESP32 (Reemplaza a Node.js)
ultima_foto_bytes = None

# Inicializamos el servidor FastAPI (Reemplaza a Express)
app = FastAPI()

# Endpoint idéntico al de tu ESP32 (No tocas el código de C++)
@app.post("/upload")
async def upload_esp32(request: Request):
    global ultima_foto_bytes
    ultima_foto_bytes = await request.body()
    print(f"📸 [SERVER] Foto de {len(ultima_foto_bytes)} bytes recibida del ESP32.")
    return Response(content="Imagen guardada exitosamente en Railway", media_type="text/plain")

# ==================================================
# 2. PROCESAMIENTO MÉDICO (V10 Segmentación)
# ==================================================
def segmentar_y_limpiar_dedo(pil_image):
    if pil_image is None: return None, None
    
    # Convertir PIL (Frontend/Memoria) a OpenCV (Backend)
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contornos:
        alto, ancho = gray.shape[:2]
        imagen_ia = gray[int(alto*0.2):int(alto*0.8), int(ancho*0.15):int(ancho*0.85)]
    else:
        contorno_dedo = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contorno_dedo)
        roi_img = gray[y:y+h, x:x+w]
        
        margen_h, margen_w = int(h * 0.05), int(w * 0.10)
        imagen_ia = roi_img[margen_h:h-margen_h, margen_w:w-margen_w]

    if imagen_ia.size == 0: return None, None

    invertida = cv2.bitwise_not(imagen_ia)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_enh = clahe.apply(invertida)
    img_enh = cv2.bilateralFilter(img_enh, 5, 65, 65)
    img_final = cv2.resize(img_enh, (300, 200))
    img_rgb = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(img_rgb), img_final

# ==================================================
# 3. CEREBRO IA COMPRIMIDO (ONNX Runtime)
# ==================================================
class ExtractorBiometricoONNX:
    def __init__(self, model_path="resnet_venas_fp16.onnx"):
        # Cargamos el archivo PDF matemático (¡No usa PyTorch!)
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def preprocess(self, pil_image):
        # Hacemos manualmente lo que hacía torchvision (Reducir RAM a 0)
        img = pil_image.resize((224, 224), Image.BILINEAR)
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_arr = img_arr.transpose(2, 0, 1) # HWC a CHW
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_arr = (img_arr - mean) / std
        return np.expand_dims(img_arr, axis=0) # Agregar batch

    def extraer_embedding(self, pil_image):
        if pil_image is None: return None
        try:
            tensor_entrada = self.preprocess(pil_image)
            # Ejecutamos la IA comprimida a la velocidad de la luz
            embedding = self.ort_session.run(None, {self.input_name: tensor_entrada})[0].flatten()
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"Error IA: {e}")
            return None

print("Cargando Motor ONNX de ultra-bajo consumo...")
extractor_ia = ExtractorBiometricoONNX()

# ==================================================
# 4. LÓGICA DE NEGOCIO (FRONTEND WEB)
# ==================================================
def obtener_foto_del_esp32():
    """Jala la foto en memoria que el ESP32 acaba de mandar al servidor"""
    global ultima_foto_bytes
    if ultima_foto_bytes is None:
        return None
    return Image.open(io.BytesIO(ultima_foto_bytes))

def registrar_nuevo_cliente(cedula, img1, img2, img3, img4, img5):
    if not cedula: return "⚠️ Error: Faltan Cédula."
    cedula_limpia = cedula.strip().upper()
    ruta_cliente = os.path.join(DATABASE_DIR, cedula_limpia)

    if os.path.exists(ruta_cliente):
        return f"⚠️ La Cédula {cedula_limpia} ya existe."

    imagenes = [img1, img2, img3, img4, img5]
    if any(img is None for img in imagenes): return "⚠️ Faltan fotos del ESP32."

    os.makedirs(ruta_cliente)
    muestras_validas = 0
    for idx, img_pil in enumerate(imagenes):
        imagen_ia_pil, _ = segmentar_y_limpiar_dedo(img_pil)
        if imagen_ia_pil is None: continue
        embedding = extractor_ia.extraer_embedding(imagen_ia_pil)
        if embedding is not None:
            np.save(os.path.join(ruta_cliente, f"muestra_{idx+1}.npy"), embedding)
            img_pil.save(os.path.join(ruta_cliente, f"original_{idx+1}.jpg"))
            muestras_validas += 1

    if muestras_validas == 5:
        return f"✅ ¡Cliente {cedula_limpia} registrado exitosamente!"
    else:
        shutil.rmtree(ruta_cliente)
        return "⚠️ Error al procesar. Repita las capturas."

def verificar_identidad(cedula, img_pil):
    if not cedula: return "⚠️ Ingrese Cédula.", None
    cedula_limpia = cedula.strip().upper()
    ruta_cliente = os.path.join(DATABASE_DIR, cedula_limpia)

    if not os.path.exists(ruta_cliente):
        return f"🔴 ALERTA: Cédula {cedula_limpia} no registrada.", None
    if img_pil is None: return "Cargue foto desde ESP32.", None
    
    imagen_ia_pil, vision_algoritmo = segmentar_y_limpiar_dedo(img_pil)
    if imagen_ia_pil is None: return "Error óptico.", None

    embedding_nuevo = extractor_ia.extraer_embedding(imagen_ia_pil)
    archivos_adn = glob.glob(os.path.join(ruta_cliente, "*.npy"))
    
    similitudes = [cosine_similarity(embedding_nuevo.reshape(1, -1), np.load(adn).reshape(1, -1))[0][0] for adn in archivos_adn]
    similitudes.sort(reverse=True)
    
    promedio_top3 = sum(similitudes[:3]) / 3 if len(similitudes) >= 3 else sum(similitudes)/max(1,len(similitudes))
    porcentaje = round(promedio_top3 * 100, 2)

    if promedio_top3 >= SIMILARITY_THRESHOLD:
        return f"🟢 IDENTIDAD CONFIRMADA\nTitular: {cedula_limpia}\nCerteza: {porcentaje}%", vision_algoritmo
    else:
        return f"🔴 FRAUDE DETECTADO\nEl dedo no pertenece a {cedula_limpia}\nSimilitud: {porcentaje}%", vision_algoritmo

# ==================================================
# 5. PÁGINA WEB PARA EL TRABAJADOR DEL BANCO
# ==================================================
with gr.Blocks(title="Terminal Fintech - ESP32", theme=gr.themes.Default()) as interfaz:
    gr.Markdown(f"## 🏦 Panel de Control POS Biométrico\n🌐 **Servidor Integrado ESP32:** Conectado directamente por puerto API.")
    
    with gr.Tabs():
        with gr.TabItem("💳 Cajas - Validar Pago"):
            with gr.Row():
                with gr.Column():
                    in_cedula_pago = gr.Textbox(label="1. Cédula del Cliente", placeholder="Ej: V-12345678")
                    btn_obtener_foto = gr.Button("2. 📸 Extraer foto actual del ESP32", variant="secondary")
                    img_verificar = gr.Image(type="pil", label="Foto de la Máquina", interactive=False)
                    btn_verificar = gr.Button("3. Autorizar Pago", variant="primary")
                with gr.Column():
                    txt_resultado = gr.Textbox(label="Respuesta de Seguridad", lines=6)
                    img_vision = gr.Image(label="Segmentación Activa")
            
            # El botón de "Extraer foto" saca la foto de la memoria sin refrescar la página
            btn_obtener_foto.click(fn=obtener_foto_del_esp32, inputs=[], outputs=img_verificar)
            btn_verificar.click(fn=verificar_identidad, inputs=[in_cedula_pago, img_verificar], outputs=[txt_resultado, img_vision])
        
        with gr.TabItem("🔐 Plataforma - Registrar Cuenta"):
            gr.Markdown("**Proceso:** Escriba la Cédula. Haga que el cliente coloque el dedo en el hardware, cuando el hardware tome la foto, pulse 'Muestra 1'. Mueva el dedo, hardware toma foto, pulse 'Muestra 2', etc.")
            in_cedula_registro = gr.Textbox(label="Número de Cédula de Identidad")
            
            with gr.Row():
                with gr.Column():
                    btn_get1 = gr.Button("Descargar Muestra 1")
                    img_reg_1 = gr.Image(type="pil", interactive=False)
                with gr.Column():
                    btn_get2 = gr.Button("Descargar Muestra 2")
                    img_reg_2 = gr.Image(type="pil", interactive=False)
                with gr.Column():
                    btn_get3 = gr.Button("Descargar Muestra 3")
                    img_reg_3 = gr.Image(type="pil", interactive=False)
            with gr.Row():
                with gr.Column():
                    btn_get4 = gr.Button("Descargar Muestra 4")
                    img_reg_4 = gr.Image(type="pil", interactive=False)
                with gr.Column():
                    btn_get5 = gr.Button("Descargar Muestra 5")
                    img_reg_5 = gr.Image(type="pil", interactive=False)
            
            btn_get1.click(fn=obtener_foto_del_esp32, outputs=img_reg_1)
            btn_get2.click(fn=obtener_foto_del_esp32, outputs=img_reg_2)
            btn_get3.click(fn=obtener_foto_del_esp32, outputs=img_reg_3)
            btn_get4.click(fn=obtener_foto_del_esp32, outputs=img_reg_4)
            btn_get5.click(fn=obtener_foto_del_esp32, outputs=img_reg_5)

            btn_registrar = gr.Button("Vincular Patrón a Cédula", variant="primary")
            txt_registro_res = gr.Textbox(label="Estado del Servidor")
            
            btn_registrar.click(
                fn=registrar_nuevo_cliente, 
                inputs=[in_cedula_registro, img_reg_1, img_reg_2, img_reg_3, img_reg_4, img_reg_5], 
                outputs=txt_registro_res
            )

# Conectamos la Interfaz web al mismo servidor donde entra el ESP32
app = gr.mount_gradio_app(app, interfaz, path="/")