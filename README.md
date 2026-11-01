# 🧬 Sistema POS Biométrico Vascular (Escáner de Venas con ESP32-CAM)

Sistema bancario y pasarela de pagos basada en autenticación biométrica vascular subcutánea (Finger Vein Biometrics) de grado médico, con soporte para terminales físicas **ESP32-CAM** e inferencia en servidor (Local con Túnel Cloudflare o Despliegue en la Nube con Railway).

---

## 🚀 Características Principales

* **Aislamiento Vascular Adaptativo:** Segmentación inteligente de la ROI del dedo inmune a fugas de luz infrarroja directa (>210) y sombras de carcasa (<22).
* **Visión Multiescala de Grado Médico:**
  * **Banco de Filtros Gabor:** 8 orientaciones y 2 escalas para detección de flujo y orientación de vasos sanguíneos.
  * **Filtro de Vesselness de Frangi / Curvatura Hessiana:** Detección de estructuras tubulares y bifurcaciones vasculares sin verse afectado por el grosor epidérmico.
* **Correlación Estructural Inmune a Falsos Positivos:**
  * Búsqueda de alineación traslacional 2D con tolerancia a micro-rotaciones ($\pm 2.5^\circ$).
  * **Separación de seguridad:** Dedos distintos (intrusos) obtienen $\approx 20\% - 30\%$; dedos auténticos obtienen $\approx 75\% - 98\%$.
  * **Umbral bancario seguro:** 60% (Consenso estricto Top-3 de 5 muestras).
* **Doble Compatibilidad:**
  * **Endpoint nativo ESP32:** `POST /upload` para recibir imágenes JPEG en crudo disparadas por el sensor infrarrojo.
  * **Interfaz de Operador (Gradio):** Interfaz web integrada en la raíz (`/`) con soporte para pegar fotos con `Ctrl + V`, arrastrar archivos o extraer del ESP32.
  * **API REST Modular:** Endpoints `/api/v1/verify` y `/api/v1/enroll`.

---

## 🛠️ Despliegue en Railway

El proyecto está 100% preparado para desplegarse automáticamente en Railway en cuanto actives tu suscripción:

1. **Start Command (`Procfile`):**
   ```bash
   web: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
   ```

2. **Variables de Entorno Recomendadas en Railway (Settings -> Variables):**
   ```env
   OMP_NUM_THREADS=4
   MKL_NUM_THREADS=4
   OPENBLAS_NUM_THREADS=2
   TORCH_HOME=/app/.cache/torch
   PYTHONUNBUFFERED=1
   ```

3. **Requisitos de Sistema (`requirements.txt`):**
   * PyTorch CPU (`torch>=2.2.0`, `torchvision>=0.17.0`) optimizado sin dependencias CUDA pesadas.
   * `fastapi`, `uvicorn`, `gradio`, `opencv-python-headless`, `scikit-learn`.

---

## 💻 Ejecución Local (PC + Túnel Cloudflare)

Para ejecutar el sistema localmente sin pagar suscripciones y permitiendo que el ESP32 se conecte desde cualquier parte del mundo:

1. **Ejecutar el lanzador rápido:**
   Doble clic en `INICIAR_SISTEMA_BIOMETRICO.bat`.
   * Inicia el servidor FastAPI en el puerto `8000`.
   * Abre automáticamente un túnel público mundial HTTPS de Cloudflare (`https://xxxx.trycloudflare.com`).

2. **Configuración en el ESP32 (`CameraWebServer.ino`):**
   ```cpp
   String serverUrl = "https://<tu-tunel>.trycloudflare.com/upload";
   ```

---

## 📂 Estructura de Datos

* `cache/latest_esp32.jpg`: Imagen más reciente recibida del ESP32.
* `database/<CEDULA>/`:
  * `sample_1.npy` ... `sample_5.npy`: Mapas vasculares 2D de alta precisión.
  * `original_1.jpg` ... `original_5.jpg`: Respaldo de imágenes infrarrojas originales.
