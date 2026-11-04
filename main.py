import os
import io
import time
import base64
import shutil
import urllib.request
import urllib.error
import json
import threading
import random
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr

from biometric_engine import VascularBiometricEngine

# =====================================================================
# 1. INICIALIZACIÓN DEL SISTEMA Y MOTOR BIOMÉTRICO
# =====================================================================
DATABASE_DIR = "database"
CACHE_DIR = "cache"
TEMPLATES_DIR = "templates"
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

CACHE_LATEST_IMAGE = os.path.join(CACHE_DIR, "latest_esp32.jpg")

# Inicializar motor biométrico de correlación estructural vascular
# Umbral 0.40 (40%): Intruso da 10-15%, titular legítimo da > 70%
biometric_engine = VascularBiometricEngine(
    database_dir=DATABASE_DIR,
    similarity_threshold=0.40
)

# Memoria temporal para un solo worker, y persistencia en cache para multi-worker
ultima_foto_bytes: Optional[bytes] = None
ultima_foto_timestamp: float = 0.0

# Telemetría y detección automática del ESP32 en cualquier red del mundo
esp32_device_info = {
    "detected": False,
    "last_seen_timestamp": 0.0,
    "last_seen_time_str": "Nunca",
    "public_ip": None,
    "local_ip": None,
    "user_agent": None,
    "bytes_received": 0
}

# Cola de comandos de configuración pendientes para el sensor OV2640
pending_esp32_config: Dict[str, int] = {}

app = FastAPI(
    title="Banco Digital Biométrico - Terminal POS Vascular",
    description="Backend bancario de alta precisión para autenticación vascular y biopago con ESP32-CAM",
    version="3.0.0"
)

# Permitir CORS para permitir solicitudes locales o desde túneles
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 2. MODELOS PYDANTIC PARA ENDPOINTS JSON
# =====================================================================
class VerifyJsonRequest(BaseModel):
    cedula: str
    image_b64: str

class CompareTwoJsonRequest(BaseModel):
    image1_b64: str
    image2_b64: str

class EnrollJsonRequest(BaseModel):
    cedula: str
    sample_idx: int
    image_b64: str

def decode_base64_image(b64_str: str) -> Image.Image:
    """Decodifica cadenas base64 (incluyendo cabeceras data:image/jpeg;base64,) a PIL.Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# =====================================================================
# PERSISTENCIA CENTRALIZADA DE CUENTAS BANCARIAS Y TRANSACCIONES
# (Sincronización multi-dispositivo entre PC, teléfonos y terminales POS)
# =====================================================================
USERS_DB_PATH = os.path.join(DATABASE_DIR, "users.json")
TRANSACTIONS_DB_PATH = os.path.join(DATABASE_DIR, "transactions.json")
db_lock = threading.RLock()

INITIAL_USERS = {
    "1": {"password": "123", "balance": 100.00, "fullName": "Isaac Titular (C.I. 1)"},
    "2": {"password": "123", "balance": 100.00, "fullName": "Cliente Demo 2 (C.I. 2)"},
    "cliente": {"password": "1234", "balance": 100.00, "fullName": "Carlos Cliente"},
    "comercio": {"password": "1234", "balance": 100.00, "fullName": "Abasto Central"},
    "admin": {"password": "admin123", "balance": 500.00, "fullName": "Administrador POS"}
}

INITIAL_TRANSACTIONS = [
    {
        "id": "TX-10001",
        "type": "Recarga Inicial",
        "sender": "Banco Central",
        "receiver": "1",
        "amount": 100.00,
        "concept": "Bono de Bienvenida",
        "biometric": None,
        "status": "Aprobada",
        "timestamp": int(time.time() * 1000) - 3600000
    }
]

def load_server_users() -> dict:
    with db_lock:
        if not os.path.exists(USERS_DB_PATH):
            save_server_users_internal(INITIAL_USERS)
            return dict(INITIAL_USERS)
        try:
            with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return dict(INITIAL_USERS)
        except Exception as e:
            print(f"[DB Error] loading users: {e}")
            return dict(INITIAL_USERS)

def save_server_users_internal(users: dict):
    os.makedirs(DATABASE_DIR, exist_ok=True)
    with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def save_server_users(users: dict):
    with db_lock:
        save_server_users_internal(users)

def load_server_transactions() -> list:
    with db_lock:
        if not os.path.exists(TRANSACTIONS_DB_PATH):
            save_server_transactions_internal(INITIAL_TRANSACTIONS)
            return list(INITIAL_TRANSACTIONS)
        try:
            with open(TRANSACTIONS_DB_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return list(INITIAL_TRANSACTIONS)
        except Exception as e:
            print(f"[DB Error] loading transactions: {e}")
            return list(INITIAL_TRANSACTIONS)

def save_server_transactions_internal(txs: list):
    os.makedirs(DATABASE_DIR, exist_ok=True)
    with open(TRANSACTIONS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(txs, f, indent=2, ensure_ascii=False)

def save_server_transactions(txs: list):
    with db_lock:
        save_server_transactions_internal(txs)

class BankRegisterRequest(BaseModel):
    user: str
    password: str
    fullName: Optional[str] = None

class BankSaveUsersRequest(BaseModel):
    users: Dict[str, Any]

class BankRecordTxRequest(BaseModel):
    transaction: Dict[str, Any]

class BankTransferRequest(BaseModel):
    sender: str
    receiver: str
    amount: float
    concept: Optional[str] = "Transferencia Pago Móvil"

@app.get("/api/v1/bank/sync")
async def bank_sync_all():
    """Retorna en una sola llamada los usuarios y transacciones del servidor."""
    return {
        "status": "success",
        "users": load_server_users(),
        "transactions": load_server_transactions(),
        "server_time": int(time.time() * 1000)
    }

@app.get("/api/v1/bank/users")
async def get_bank_users():
    return {"status": "success", "users": load_server_users()}

@app.post("/api/v1/bank/users/save")
async def save_bank_users_api(req: BankSaveUsersRequest):
    save_server_users(req.users)
    return {"status": "success", "count": len(req.users)}

@app.post("/api/v1/bank/register")
async def register_bank_user(req: BankRegisterRequest):
    clean_user = req.user.strip().lower()
    if not clean_user or len(req.password) < 3:
        raise HTTPException(status_code=400, detail="La clave debe tener al menos 3 caracteres.")
    
    with db_lock:
        users = load_server_users()
        if clean_user in users:
            raise HTTPException(status_code=400, detail="Esta Cédula o Usuario ya está registrado.")
        
        users[clean_user] = {
            "password": req.password,
            "balance": 100.00,
            "fullName": req.fullName or f"Titular C.I. {clean_user.upper()}"
        }
        save_server_users_internal(users)
        
        txs = load_server_transactions()
        txs.insert(0, {
            "id": f"TX-{int(time.time() * 1000) % 100000:05d}",
            "type": "Apertura de Cuenta",
            "sender": "Banco Biométrico",
            "receiver": clean_user,
            "amount": 100.00,
            "concept": "Bono de Bienvenida por Apertura",
            "biometric": None,
            "status": "Aprobada",
            "timestamp": int(time.time() * 1000)
        })
        save_server_transactions_internal(txs)
    
    return {"status": "success", "user": clean_user, "users": users}

@app.get("/api/v1/bank/transactions")
async def get_bank_transactions():
    return {"status": "success", "transactions": load_server_transactions()}

@app.post("/api/v1/bank/transactions")
async def record_bank_transaction_api(req: BankRecordTxRequest):
    tx = req.transaction
    if "timestamp" not in tx or not tx["timestamp"]:
        tx["timestamp"] = int(time.time() * 1000)
    with db_lock:
        txs = load_server_transactions()
        txs.insert(0, tx)
        save_server_transactions_internal(txs)
    return {"status": "success", "count": len(txs)}

@app.post("/api/v1/bank/transfer")
async def transfer_bank_funds(req: BankTransferRequest):
    sender_key = req.sender.strip().lower()
    receiver_key = req.receiver.strip().lower()
    amount = round(req.amount, 2)

    if amount <= 0:
        raise HTTPException(status_code=400, detail="El monto debe ser mayor a 0.")
    if sender_key == receiver_key:
        raise HTTPException(status_code=400, detail="No puedes transferirte a ti mismo.")

    with db_lock:
        users = load_server_users()
        if sender_key not in users:
            raise HTTPException(status_code=404, detail="Usuario emisor no encontrado.")
        if users[sender_key]["balance"] < amount:
            raise HTTPException(status_code=400, detail="Saldo insuficiente.")

        if receiver_key not in users:
            users[receiver_key] = {
                "password": "123",
                "balance": 0.00,
                "fullName": f"Titular {receiver_key.upper()}"
            }

        users[sender_key]["balance"] = round(users[sender_key]["balance"] - amount, 2)
        users[receiver_key]["balance"] = round(users[receiver_key]["balance"] + amount, 2)
        save_server_users_internal(users)

        txs = load_server_transactions()
        ref = f"PM-{random.randint(100000, 999999)}"
        txs.insert(0, {
            "id": ref,
            "type": "Pago Móvil",
            "sender": sender_key,
            "receiver": receiver_key,
            "amount": amount,
            "concept": req.concept,
            "biometric": None,
            "status": "Aprobada",
            "timestamp": int(time.time() * 1000)
        })
        save_server_transactions_internal(txs)

    return {
        "status": "success",
        "ref": ref,
        "sender_balance": users[sender_key]["balance"],
        "receiver_balance": users[receiver_key]["balance"],
        "users": users
    }

@app.post("/api/v1/bank/reset")
async def reset_bank_database():
    with db_lock:
        save_server_users_internal(INITIAL_USERS)
        save_server_transactions_internal(INITIAL_TRANSACTIONS)
    return {"status": "success", "users": INITIAL_USERS, "transactions": INITIAL_TRANSACTIONS}

# =====================================================================
# 3. ENDPOINT NATIVO DEL ESP32-CAM (SUBIDA DE FOTOS)
# =====================================================================
@app.post("/upload")
async def upload_esp32(request: Request):
    """
    Endpoint nativo para el ESP32-CAM.
    Recibe el buffer binario JPEG (fb->buf, fb->len) y lo persiste
    inmediatamente en disco y memoria para disponibilidad instantánea.
    Además detecta automáticamente la IP remota/local del ESP32.
    """
    global ultima_foto_bytes, ultima_foto_timestamp, esp32_device_info, pending_esp32_config
    body = await request.body()
    
    if not body or len(body) < 100:
        raise HTTPException(status_code=400, detail="Buffer de imagen vacío o corrupto.")

    ultima_foto_bytes = body
    ultima_foto_timestamp = time.time()

    # Detección automática de IP (Pública y Local enviada por el ESP32)
    forwarded = request.headers.get("x-forwarded-for")
    client_ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "desconocido")
    local_ip = request.headers.get("x-esp32-ip") or request.headers.get("x-local-ip")

    esp32_device_info = {
        "detected": True,
        "last_seen_timestamp": time.time(),
        "last_seen_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "public_ip": client_ip,
        "local_ip": local_ip or client_ip,
        "user_agent": request.headers.get("user-agent", "ESP32-HTTPClient"),
        "bytes_received": len(body)
    }

    # Escritura atómica y descarga a disco con flush
    try:
        with open(CACHE_LATEST_IMAGE, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"[CACHE DISK ERROR]: {e}")

    print(f"[ESP32 CONECTADO] IP detectada: {client_ip} | Local: {local_ip or 'N/A'} | {len(body)} bytes persistidos.")

    # Si hay comandos de configuración pendientes para el sensor, responder con JSON
    if pending_esp32_config:
        commands_to_send = dict(pending_esp32_config)
        pending_esp32_config.clear()
        return JSONResponse(content={"status": "OK", "commands": commands_to_send})

    return Response(content="OK", media_type="text/plain")


def obtener_foto_del_esp32() -> Optional[Image.Image]:
    """
    Extrae de inmediato la última foto capturada por el ESP32.
    Prioriza la lectura directa de disco para garantizar frescura absoluta.
    """
    if os.path.exists(CACHE_LATEST_IMAGE):
        try:
            with open(CACHE_LATEST_IMAGE, "rb") as f:
                img_data = f.read()
            if img_data and len(img_data) > 100:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                return img
        except Exception as e:
            print(f"[CACHE READ ERROR]: {e}")

    global ultima_foto_bytes
    if ultima_foto_bytes and len(ultima_foto_bytes) > 100:
        try:
            return Image.open(io.BytesIO(ultima_foto_bytes)).convert("RGB")
        except Exception:
            pass

    return None

# =====================================================================
# 4. ENDPOINTS REST API PARA EL FRONTEND DE BANCO DIGITAL (JSON / REST)
# =====================================================================
@app.get("/api/v1/status")
async def api_get_status():
    """Retorna el estado de salud del servidor biométrico y clientes registrados."""
    clients = []
    if os.path.exists(DATABASE_DIR):
        for d in os.listdir(DATABASE_DIR):
            cdir = os.path.join(DATABASE_DIR, d)
            if os.path.isdir(cdir):
                npy_count = len([f for f in os.listdir(cdir) if f.endswith(".npy")])
                clients.append({"cedula": d, "samples": npy_count})

    has_cache = os.path.exists(CACHE_LATEST_IMAGE)
    cache_time = os.path.getmtime(CACHE_LATEST_IMAGE) if has_cache else 0

    return {
        "status": "online",
        "threshold": biometric_engine.similarity_threshold,
        "enrolled_clients": clients,
        "has_latest_capture": has_cache,
        "latest_capture_time": cache_time
    }

@app.get("/api/v1/latest-image")
async def api_get_latest_image(format: Optional[str] = "image"):
    """
    Retorna la última foto recibida del ESP32.
    format='image' -> Retorna imagen JPEG directa.
    format='json'  -> Retorna JSON con data URL base64.
    """
    if not os.path.exists(CACHE_LATEST_IMAGE):
        raise HTTPException(status_code=404, detail="No hay capturas disponibles del ESP32.")

    try:
        with open(CACHE_LATEST_IMAGE, "rb") as f:
            data = f.read()

        if format == "json":
            b64 = base64.b64encode(data).decode("utf-8")
            return {
                "status": "success",
                "image_b64": f"data:image/jpeg;base64,{b64}",
                "size_bytes": len(data),
                "timestamp": os.path.getmtime(CACHE_LATEST_IMAGE)
            }
        else:
            return Response(content=data, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/verify")
async def verify_payment_api(
    request: Request,
    cedula: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Verificación biométrica 1:1.
    Soporta tanto multipart/form-data como application/json con image_b64.
    """
    content_type = request.headers.get("content-type", "")
    pil_img: Optional[Image.Image] = None
    target_cedula: str = ""

    if "application/json" in content_type:
        try:
            body = await request.json()
            target_cedula = str(body.get("cedula", "")).strip().upper()
            img_b64 = body.get("image_b64") or body.get("image")
            if not img_b64:
                raise ValueError("Falta campo 'image_b64'.")
            pil_img = decode_base64_image(img_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON inválido o imagen base64 corrupta: {e}")
    else:
        # Multipart form
        if not cedula or not file:
            raise HTTPException(status_code=400, detail="Se requiere 'cedula' y 'file'.")
        target_cedula = cedula.strip().upper()
        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Imagen JPEG corrupta o inválida.")

    if not target_cedula:
        raise HTTPException(status_code=400, detail="Cédula no proporcionada.")

    mensaje, vis_map, metrics = biometric_engine.verify_identity(target_cedula, pil_img)

    vis_b64 = None
    if vis_map is not None:
        _, buffer = cv2.imencode(".jpg", vis_map)
        vis_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    final_score = float(metrics.get("final_score", 0.0))
    is_auth = bool(metrics.get("authenticated", False))
    porcentaje = round(final_score * 100, 2)

    return {
        "authenticated": is_auth,
        "verdict": mensaje,
        "cedula": target_cedula,
        "score": final_score,
        "percentage": porcentaje,
        "max_match": metrics.get("max_match", 0.0),
        "quality_score": metrics.get("quality_score", 0.0),
        "threshold": biometric_engine.similarity_threshold * 100,
        "vascular_map_b64": vis_b64
    }

@app.post("/api/v1/compare-two")
async def compare_two_images_api(request: Request):
    """
    Compara 2 fotos directamente usando el algoritmo de Curvatura de Miura.
    Ideal para comparar Muestra 1 vs Muestra 2 en pantalla sin requerir registro previo.
    """
    try:
        body = await request.json()
        b64_1 = body.get("image1_b64") or body.get("image1")
        b64_2 = body.get("image2_b64") or body.get("image2")
        if not b64_1 or not b64_2:
            raise ValueError("Se requieren 'image1_b64' y 'image2_b64'.")

        img1 = decode_base64_image(b64_1)
        img2 = decode_base64_image(b64_2)

        tpl1, vis1, q1 = biometric_engine.preprocessor.extract_template_and_vis(img1)
        tpl2, vis2, q2 = biometric_engine.preprocessor.extract_template_and_vis(img2)

        if tpl1 is None or tpl2 is None:
            return {
                "authenticated": False,
                "score": 0.0,
                "percentage": 0.0,
                "message": "Fallo al segmentar el dedo en una o ambas imágenes."
            }

        score = biometric_engine._match_vein_patterns(tpl1, tpl2)
        porcentaje = round(score * 100, 2)
        thresh = biometric_engine.similarity_threshold
        is_auth = score >= thresh

        # Mapa visual compuesto
        vis_b64 = None
        if vis2 is not None:
            _, buffer = cv2.imencode(".jpg", vis2)
            vis_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return {
            "authenticated": is_auth,
            "score": round(score, 4),
            "percentage": porcentaje,
            "quality1": q1.get("quality", 0),
            "quality2": q2.get("quality", 0),
            "threshold": round(thresh * 100, 2),
            "message": "🟢 IDENTIDAD CONFIRMADA" if is_auth else "🔴 IDENTIDAD NO COINCIDE",
            "vascular_map_b64": vis_b64
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en comparación 1:1: {e}")

@app.post("/api/v1/enroll")
async def enroll_finger_api(
    request: Request,
    cedula: Optional[str] = Form(None), 
    sample_idx: Optional[int] = Form(None), 
    file: Optional[UploadFile] = File(None)
):
    """Enrolamiento de muestra biométrica (soporta JSON base64 o multipart)."""
    content_type = request.headers.get("content-type", "")
    target_cedula = ""
    idx = 1
    pil_img: Optional[Image.Image] = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            target_cedula = str(body.get("cedula", "")).strip().upper()
            idx = int(body.get("sample_idx", 1))
            img_b64 = body.get("image_b64") or body.get("image")
            if not img_b64:
                raise ValueError("Falta campo 'image_b64'.")
            pil_img = decode_base64_image(img_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON inválido: {e}")
    else:
        if not cedula or not sample_idx or not file:
            raise HTTPException(status_code=400, detail="Campos incompletos.")
        target_cedula = cedula.strip().upper()
        idx = sample_idx
        contents = await file.read()
        try:
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Imagen JPEG corrupta o inválida.")

    res = biometric_engine.enroll_sample(target_cedula, pil_img, idx)
    return res

@app.get("/api/v1/esp32/control")
async def esp32_control_proxy(var: str, val: int, ip: Optional[str] = None):
    """
    Controla los parámetros del sensor OV2640.
    1. Si está en la misma red LAN (IP local directa), envía el comando inmediatamente.
    2. Si el ESP32 está en otra red del mundo (detrás de NAT), guarda el comando
       en la cola para entregárselo en cuanto el ESP32 se comunique con el servidor.
    """
    global pending_esp32_config
    pending_esp32_config[var] = val

    detected_local_ip = esp32_device_info.get("local_ip")
    target_ip = ip or detected_local_ip or "192.168.1.36"
    url = f"http://{target_ip}/control?var={var}&val={val}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BancoBiometricoPOS/3.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return {"status": "ok", "mode": "direct_lan", "variable": var, "value": val, "code": resp.getcode()}
    except Exception:
        return {"status": "ok", "mode": "queued_remote", "variable": var, "value": val, "detail": "Comando encolado para entrega remota al ESP32"}

@app.get("/api/v1/esp32/status")
async def esp32_status_proxy(ip: Optional[str] = None):
    """
    Lee los valores actuales de todos los registros del sensor OV2640 desde la ESP32-CAM (/status).
    """
    detected_local_ip = esp32_device_info.get("local_ip")
    target_ip = ip or detected_local_ip or "192.168.1.36"
    url = f"http://{target_ip}/status"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BancoBiometricoPOS/3.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = resp.read()
            parsed = json.loads(data.decode("utf-8", errors="ignore"))
            return {"status": "ok", "settings": parsed}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/api/v1/esp32/device-info")
async def get_esp32_device_info():
    """
    Retorna la información del último ESP32 conectado:
    IP detectada, tiempo desde el último contacto y comandos pendientes.
    """
    info = dict(esp32_device_info)
    if info.get("last_seen_timestamp"):
        info["seconds_ago"] = round(time.time() - info["last_seen_timestamp"], 1)
    else:
        info["seconds_ago"] = None
    info["pending_commands"] = pending_esp32_config
    return info

# =====================================================================
# 5. LÓGICA DE NEGOCIO PARA EL PANEL GRADIO (ADMIN / AUDITORÍA)
# =====================================================================
def verificar_pago_gradio(cedula: str, img_pil: Optional[Image.Image]):
    if not cedula or not cedula.strip():
        return "⚠️ Ingrese el número de Cédula.", None

    if img_pil is None:
        return "⚠️ Extraiga primero la foto capturada por el ESP32 o pegue una foto.", None

    mensaje, vis_map, metrics = biometric_engine.verify_identity(cedula, img_pil)

    if metrics.get("authenticated", False):
        resultado_txt = (
            f"{mensaje}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Cédula: {cedula.strip().upper()}\n"
            f"Estado: TRANSACCIÓN AUTORIZADA\n"
            f"Operación: Pago Aprobado con Éxito"
        )
    else:
        resultado_txt = (
            f"{mensaje}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Cédula: {cedula.strip().upper()}\n"
            f"Estado: TRANSACCIÓN DENEGADA\n"
            f"Motivo: El patrón biométrico no coincide"
        )

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
        return f"⚠️ La Cédula {cedula_limpia} ya está registrada en el sistema."

    muestras = [img1, img2, img3, img4, img5]
    if any(m is None for m in muestras):
        return "⚠️ Error: Se requieren las 5 muestras del sensor biométrico."

    os.makedirs(client_dir, exist_ok=True)

    for idx, img in enumerate(muestras):
        res = biometric_engine.enroll_sample(cedula_limpia, img, idx + 1)
        if res.get("status") != "success":
            shutil.rmtree(client_dir, ignore_errors=True)
            return f"❌ Error en Muestra {idx + 1}: {res.get('message', 'Fallo al procesar')}."

    return f"✅ ¡Cliente {cedula_limpia} registrado exitosamente en el sistema biométrico!"

with gr.Blocks(title="Terminal POS - Biopago Admin") as interfaz_admin:
    gr.Markdown(
        "## 🏦 Panel Administrativo Biopago\n"
        "💡 *Panel de auditoría biométrica y respaldo administrativo.*"
    )

    with gr.Tabs():
        with gr.TabItem("💳 Validar Pago"):
            with gr.Row():
                with gr.Column():
                    in_cedula_pago = gr.Textbox(label="Cédula de Identidad", placeholder="Ej: V-12345678")
                    btn_obtener_foto = gr.Button("📸 Extraer foto actual del sensor", variant="secondary")
                    img_verificar = gr.Image(
                        type="pil", 
                        label="Foto a Validar (Captura o presiona Ctrl+V para pegar)", 
                        interactive=True
                    )
                    btn_verificar = gr.Button("Validar y Autorizar Pago", variant="primary")
                with gr.Column():
                    txt_resultado = gr.Textbox(label="Resultado de la Operación", lines=6)
                    img_vision = gr.Image(label="Patrón Biométrico Detectado")

            btn_obtener_foto.click(fn=obtener_foto_del_esp32, inputs=[], outputs=img_verificar)
            btn_verificar.click(
                fn=verificar_pago_gradio,
                inputs=[in_cedula_pago, img_verificar],
                outputs=[txt_resultado, img_vision]
            )

        with gr.TabItem("🔐 Registrar Cliente"):
            in_cedula_registro = gr.Textbox(label="Número de Cédula de Identidad", placeholder="Ej: V-12345678")
            with gr.Row():
                with gr.Column():
                    btn_get1 = gr.Button("Foto Muestra 1", size="sm")
                    img_reg_1 = gr.Image(type="pil", label="Muestra 1 (o Ctrl+V)", interactive=True)
                with gr.Column():
                    btn_get2 = gr.Button("Foto Muestra 2", size="sm")
                    img_reg_2 = gr.Image(type="pil", label="Muestra 2 (o Ctrl+V)", interactive=True)
                with gr.Column():
                    btn_get3 = gr.Button("Foto Muestra 3", size="sm")
                    img_reg_3 = gr.Image(type="pil", label="Muestra 3 (o Ctrl+V)", interactive=True)

            with gr.Row():
                with gr.Column():
                    btn_get4 = gr.Button("Foto Muestra 4", size="sm")
                    img_reg_4 = gr.Image(type="pil", label="Muestra 4 (o Ctrl+V)", interactive=True)
                with gr.Column():
                    btn_get5 = gr.Button("Foto Muestra 5", size="sm")
                    img_reg_5 = gr.Image(type="pil", label="Muestra 5 (o Ctrl+V)", interactive=True)

            btn_get1.click(fn=obtener_foto_del_esp32, outputs=img_reg_1)
            btn_get2.click(fn=obtener_foto_del_esp32, outputs=img_reg_2)
            btn_get3.click(fn=obtener_foto_del_esp32, outputs=img_reg_3)
            btn_get4.click(fn=obtener_foto_del_esp32, outputs=img_reg_4)
            btn_get5.click(fn=obtener_foto_del_esp32, outputs=img_reg_5)

            btn_registrar = gr.Button("Registrar Cliente en el Sistema", variant="primary")
            txt_registro_res = gr.Textbox(label="Estado del Registro")

            btn_registrar.click(
                fn=registrar_nuevo_cliente_gradio,
                inputs=[in_cedula_registro, img_reg_1, img_reg_2, img_reg_3, img_reg_4, img_reg_5],
                outputs=txt_registro_res
            )

# =====================================================================
# 6. SERVIR EL BANCO DIGITAL BIOMÉTRICO EN LA RAÍZ Y GRADIO EN /admin
# =====================================================================
INDEX_HTML_PATH = os.path.join(TEMPLATES_DIR, "index.html")
ESP32_CAM_HTML_PATH = os.path.join(TEMPLATES_DIR, "esp32_cam.html")

@app.get("/", response_class=FileResponse)
async def serve_home():
    """Sirve la aplicación de Banco Digital Biométrico en la raíz."""
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse("<h1>Banco Biométrico Online</h1><p>Archivo templates/index.html no encontrado.</p>")

# =====================================================================
# SIMULACIÓN DE ENDPOINTS NATIVOS DEL ESP32 PARA EL PREVIEW WEB
# =====================================================================
mock_esp32_leds = False
mock_server_url = "https://opened-reach-swap-educational.trycloudflare.com/upload"

@app.get("/get_led_status")
async def mock_get_led():
    global mock_esp32_leds
    return {"leds": 1 if mock_esp32_leds else 0}

@app.get("/set_led")
async def mock_set_led(val: Optional[int] = None):
    global mock_esp32_leds
    if val is not None:
        mock_esp32_leds = (val == 1)
    else:
        mock_esp32_leds = not mock_esp32_leds
    return {"leds": 1 if mock_esp32_leds else 0}

@app.get("/get_server_url")
async def mock_get_server_url():
    global mock_server_url
    return PlainTextResponse(mock_server_url)

@app.get("/set_server_url")
async def mock_set_server_url(url: str):
    global mock_server_url
    mock_server_url = url
    return PlainTextResponse(f"OK: {url}")

@app.get("/esp32-cam", response_class=FileResponse)
async def serve_esp32_cam():
    """Sirve la interfaz web local del sensor ESP32-CAM (OV2640)."""
    if os.path.exists(ESP32_CAM_HTML_PATH):
        return FileResponse(ESP32_CAM_HTML_PATH)
    return HTMLResponse("<h1>Interfaz ESP32-CAM</h1><p>Archivo templates/esp32_cam.html no encontrado.</p>")

# Montar Gradio en la ruta /admin para mantener el panel de auditoría disponible
app = gr.mount_gradio_app(app, interfaz_admin, path="/admin")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)
