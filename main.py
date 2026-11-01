import os
import io
import base64
import shutil
import numpy as np
import cv2
from PIL import Image
from typing import Optional

from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException
import gradio as gr

from biometric_engine import VascularBiometricEngine

# =====================================================================
# 1. INICIALIZACIÓN DEL SISTEMA Y MOTOR BIOMÉTRICO
# =====================================================================
DATABASE_DIR = "database"
CACHE_DIR = "cache"
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_LATEST_IMAGE = os.path.join(CACHE_DIR, "latest_esp32.jpg")

# Inicializar motor biométrico de correlación estructural vascular
# Umbral 0.60 (60%): Intruso da 20-30%, titular legítimo da > 75%
biometric_engine = VascularBiometricEngine(
    database_dir=DATABASE_DIR,
    similarity_threshold=0.60
)

# Memoria temporal para un solo worker, y persistencia en cache para multi-worker
ultima_foto_bytes: Optional[bytes] = None

app = FastAPI(
    title="Terminal POS Biométrica - Escáner de Venas FP32",
    description="Backend de alta precisión para autenticación vascular con ESP32-CAM",
    version="2.0.0"
)

# =====================================================================
# 2. ENDPOINT NATIVO DEL ESP32-CAM (SUBIDA DE FOTOS)
# =====================================================================
@app.post("/upload")
async def upload_esp32(request: Request):
    """
    Endpoint idéntico al del firmware Arduino del ESP32-CAM.
    Recibe el buffer binario JPEG en crudo y lo almacena tanto en memoria
    como en caché de disco para compatibilidad multi-proceso de Uvicorn.
    """
    global ultima_foto_bytes
    ultima_foto_bytes = await request.body()
    
    if not ultima_foto_bytes:
        raise HTTPException(status_code=400, detail="Buffer de imagen vacío.")

    # Guardar en caché local para que cualquier worker de Uvicorn pueda accederlo
    try:
        with open(CACHE_LATEST_IMAGE, "wb") as f:
            f.write(ultima_foto_bytes)
    except Exception as e:
        print(f"[CACHE ERROR]: No se pudo escribir en disco: {e}")

    print(f"[ESP32] Foto de {len(ultima_foto_bytes)} bytes recibida y almacenada.")
    return Response(content="Imagen recibida exitosamente por el servidor biométrico", media_type="text/plain")


def obtener_foto_del_esp32() -> Optional[Image.Image]:
    """Recupera la última foto capturada por el ESP32 desde memoria o disco."""
    global ultima_foto_bytes
    if ultima_foto_bytes:
        try:
            return Image.open(io.BytesIO(ultima_foto_bytes)).convert("RGB")
        except Exception:
            pass

    if os.path.exists(CACHE_LATEST_IMAGE):
        try:
            return Image.open(CACHE_LATEST_IMAGE).convert("RGB")
        except Exception:
            pass

    return None

# =====================================================================
# 3. ENDPOINTS API REST (PARA APPS MÓVILES / CLIENTES EXTERNOS)
# =====================================================================
@app.post("/api/v1/verify")
async def verify_payment_api(cedula: str = Form(...), file: UploadFile = File(...)):
    """Verificación 1:1 mediante API REST multipart."""
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen JPEG corrupta o inválida.")

    mensaje, vis_map, metrics = biometric_engine.verify_identity(cedula, pil_img)

    vis_b64 = None
    if vis_map is not None:
        _, buffer = cv2.imencode(".jpg", vis_map)
        vis_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "verdict": mensaje,
        "metrics": metrics,
        "vascular_map_b64": vis_b64
    }


@app.post("/api/v1/enroll")
async def enroll_finger_api(
    cedula: str = Form(...), 
    sample_idx: int = Form(...), 
    file: UploadFile = File(...)
):
    """Enrolamiento individual de muestra mediante API REST."""
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Imagen JPEG corrupta o inválida.")

    return biometric_engine.enroll_sample(cedula, pil_img, sample_idx)

# =====================================================================
# 4. LÓGICA DE NEGOCIO PARA EL PANEL WEB DEL CAJERO (GRADIO)
# =====================================================================
def verificar_pago_gradio(cedula: str, img_pil: Optional[Image.Image]):
    if not cedula or not cedula.strip():
        return "⚠️ Ingrese el número de Cédula.", None

    if img_pil is None:
        return "⚠️ Extraiga primero la foto capturada por el ESP32.", None

    mensaje, vis_map, metrics = biometric_engine.verify_identity(cedula, img_pil)

    if metrics.get("authenticated", False):
        resultado_txt = (
            f"{mensaje}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Titular: {cedula.strip().upper()}\n"
            f"Score de Coincidencia: {metrics.get('final_score', 0)*100:.2f}%\n"
            f"Mejor Muestra Individual: {metrics.get('max_match', 0)*100:.2f}%\n"
            f"Calidad Vascular IR: {metrics.get('quality_score', 0)}%\n"
            f"Estado: TRANSACCIÓN AUTORIZADA"
        )
    else:
        resultado_txt = (
            f"{mensaje}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Cédula consultada: {cedula.strip().upper()}\n"
            f"Score obtenido: {metrics.get('final_score', 0)*100:.2f}% "
            f"(Mínimo requerido: {biometric_engine.similarity_threshold*100:.0f}%)\n"
            f"Mejor coincidencia: {metrics.get('max_match', 0)*100:.2f}%\n"
            f"Calidad Vascular IR: {metrics.get('quality_score', 0)}%\n"
            f"Estado: TRANSACCIÓN RECHAZADA (Dedo no coincide)"
        )

    # Convertir mapa de visualización BGR a RGB para Gradio
    img_display = None
    if vis_map is not None:
        if len(vis_map.shape) == 2:
            img_display = cv2.cvtColor(vis_map, cv2.COLOR_GRAY2RGB)
        else:
            img_display = cv2.cvtColor(vis_map, cv2.COLOR_BGR2RGB)

    return resultado_txt, img_display


def registrar_nuevo_cliente_gradio(cedula: str, img1, img2, img3, img4, img5):
    if not cedula or not cedula.strip():
        return "⚠️ Error: Falta ingresar el número de Cédula."

    cedula_limpia = cedula.strip().upper()
    client_dir = os.path.join(DATABASE_DIR, cedula_limpia)

    if os.path.exists(client_dir) and len(os.listdir(client_dir)) > 0:
        return f"⚠️ La Cédula {cedula_limpia} ya está registrada en la base de datos."

    muestras = [img1, img2, img3, img4, img5]
    if any(m is None for m in muestras):
        return "⚠️ Error: Se requieren las 5 muestras completas del sensor ESP32."

    os.makedirs(client_dir, exist_ok=True)
    reporte_calidades = []

    for idx, img in enumerate(muestras):
        res = biometric_engine.enroll_sample(cedula_limpia, img, idx + 1)
        if res.get("status") != "success":
            shutil.rmtree(client_dir, ignore_errors=True)
            return f"❌ Error en Muestra {idx + 1}: {res.get('message', 'Fallo al procesar')}."

        # Guardar foto original de respaldo
        img.save(os.path.join(client_dir, f"original_{idx + 1}.jpg"))
        reporte_calidades.append(f"M{idx + 1}: {res.get('quality', 0)}%")

    calidades_str = " | ".join(reporte_calidades)
    return f"✅ ¡Cliente {cedula_limpia} registrado exitosamente con ConvNeXt FP32!\nCalidades vasculares: [{calidades_str}]"


# =====================================================================
# 5. INTERFAZ GRÁFICA DE USUARIO (GRADIO)
# =====================================================================
with gr.Blocks(title="Terminal Fintech POS - ConvNeXt FP32") as interfaz:
    gr.Markdown(
        "## 🏦 Panel de Control POS Biométrico Vascular\n"
        "🌐 **Motor Deep Learning:** ConvNeXt-Small (FP32) + SpatialPyramidGeM + Frangi/Gabor.\n"
        "💡 *Puedes recibir fotos del ESP32 o pegar imágenes directamente con **Ctrl + V** / subir archivo.*"
    )

    with gr.Tabs():
        with gr.TabItem("💳 Cajas - Validar Pago"):
            with gr.Row():
                with gr.Column():
                    in_cedula_pago = gr.Textbox(label="1. Cédula de Identidad", placeholder="Ej: V-12345678")
                    btn_obtener_foto = gr.Button("📸 Extraer foto actual del ESP32", variant="secondary")
                    img_verificar = gr.Image(
                        type="pil", 
                        label="Foto a Validar (ESP32 o Haz clic y presiona Ctrl+V para pegar)", 
                        interactive=True
                    )
                    btn_verificar = gr.Button("Validar y Autorizar Pago", variant="primary")
                with gr.Column():
                    txt_resultado = gr.Textbox(label="Respuesta de Seguridad Bancaria", lines=8)
                    img_vision = gr.Image(label="Segmentación y Mapa Vascular (Frangi + Gabor)")

            btn_obtener_foto.click(fn=obtener_foto_del_esp32, inputs=[], outputs=img_verificar)
            btn_verificar.click(
                fn=verificar_pago_gradio,
                inputs=[in_cedula_pago, img_verificar],
                outputs=[txt_resultado, img_vision]
            )

        with gr.TabItem("🔐 Plataforma - Registrar Cliente"):
            gr.Markdown(
                "**Procedimiento:**\n"
                "1. Ingrese la Cédula.\n"
                "2. Cargue las 5 muestras del dedo (puedes pegarlas con **Ctrl + V**, arrastrar fotos desde tu PC, o usar el botón del ESP32).\n"
                "3. Presione 'Vincular Patrón Vascular a Cédula'."
            )
            in_cedula_registro = gr.Textbox(label="Número de Cédula de Identidad", placeholder="Ej: V-12345678")

            with gr.Row():
                with gr.Column():
                    btn_get1 = gr.Button("Descargar ESP32 M1", size="sm")
                    img_reg_1 = gr.Image(type="pil", label="Muestra 1 (Pega con Ctrl+V)", interactive=True)
                with gr.Column():
                    btn_get2 = gr.Button("Descargar ESP32 M2", size="sm")
                    img_reg_2 = gr.Image(type="pil", label="Muestra 2 (Pega con Ctrl+V)", interactive=True)
                with gr.Column():
                    btn_get3 = gr.Button("Descargar ESP32 M3", size="sm")
                    img_reg_3 = gr.Image(type="pil", label="Muestra 3 (Pega con Ctrl+V)", interactive=True)

            with gr.Row():
                with gr.Column():
                    btn_get4 = gr.Button("Descargar ESP32 M4", size="sm")
                    img_reg_4 = gr.Image(type="pil", label="Muestra 4 (Pega con Ctrl+V)", interactive=True)
                with gr.Column():
                    btn_get5 = gr.Button("Descargar ESP32 M5", size="sm")
                    img_reg_5 = gr.Image(type="pil", label="Muestra 5 (Pega con Ctrl+V)", interactive=True)

            btn_get1.click(fn=obtener_foto_del_esp32, outputs=img_reg_1)
            btn_get2.click(fn=obtener_foto_del_esp32, outputs=img_reg_2)
            btn_get3.click(fn=obtener_foto_del_esp32, outputs=img_reg_3)
            btn_get4.click(fn=obtener_foto_del_esp32, outputs=img_reg_4)
            btn_get5.click(fn=obtener_foto_del_esp32, outputs=img_reg_5)

            btn_registrar = gr.Button("Vincular Patrón Vascular a Cédula", variant="primary")
            txt_registro_res = gr.Textbox(label="Estado del Servidor")

            btn_registrar.click(
                fn=registrar_nuevo_cliente_gradio,
                inputs=[in_cedula_registro, img_reg_1, img_reg_2, img_reg_3, img_reg_4, img_reg_5],
                outputs=txt_registro_res
            )

# Montar la interfaz web interactiva en la raíz del servidor FastAPI
app = gr.mount_gradio_app(app, interfaz, path="/")
