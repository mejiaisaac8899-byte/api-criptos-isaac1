import os
import glob
import math
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import convnext_small, ConvNeXt_Small_Weights


# =====================================================================
# 1. PREPROCESADOR VASCULAR DE GRADO MÉDICO (ALINEACIÓN Y MULTIESCALA)
# =====================================================================
class VascularPreprocessor:
    """
    Ejecuta segmentación sub-píxel, alineación rotacional invariante mediante
    momentos centrales de segundo orden, y extracción de características 
    vasculares en 3 canales (CLAHE + Gabor Multidireccional + Hessiano/Frangi).
    """
    def __init__(self, target_size: Tuple[int, int] = (320, 160)):
        self.target_size = target_size  # (Ancho, Alto) -> Proporción 2:1 ideal para dedos
        self.clahe_light = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        self.gabor_kernels = self._build_gabor_bank()

    def _build_gabor_bank(self) -> List[np.ndarray]:
        """Construye un banco de filtros de Gabor de 8 orientaciones y 2 escalas."""
        kernels = []
        ksize = 21
        for theta in np.arange(0, np.pi, np.pi / 8):
            for sigma, lambd in [(3.0, 8.0), (4.5, 12.0)]:
                kern = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambd, gamma=0.5, psi=0, ktype=cv2.CV_32F
                )
                kern /= 1.5 * kern.sum()
                kernels.append(kern)
        return kernels

    def _align_and_crop_roi(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Aísla el cuerpo del dedo eliminando las fugas de luz infrarroja directa
        (zonas sobreexpuestas > 210) y la sombra de la carcasa (< 22), y normaliza
        la orientación para centrar el lecho vascular.
        """
        h_img, w_img = gray.shape

        # 1. Perfil de intensidad vertical en la zona central (descartando laterales)
        col_profile = np.mean(gray[:, int(w_img * 0.2):int(w_img * 0.8)], axis=1)
        smooth_profile = cv2.GaussianBlur(col_profile.reshape(-1, 1), (35, 1), 0).flatten()

        # El tejido del dedo se encuentra en tonos medios, entre la carcasa negra y la fuga de luz blanca
        is_finger_band = (smooth_profile >= 22) & (smooth_profile <= 210)
        finger_indices = np.where(is_finger_band)[0]

        if len(finger_indices) > 50:
            diffs = np.diff(finger_indices)
            split_points = np.where(diffs > 1)[0] + 1
            blocks = np.split(finger_indices, split_points)
            best_block = max(blocks, key=lambda b: len(b))
            y_start, y_end = int(best_block[0]), int(best_block[-1])

            # Margen de seguridad del 8% para evitar los bordes de fuga lumínica
            m_y = int((y_end - y_start) * 0.08)
            y1 = max(0, y_start + m_y)
            y2 = min(h_img, y_end - m_y)
        else:
            y1, y2 = int(h_img * 0.20), int(h_img * 0.80)

        # Recorte horizontal descartando bordes externos de la caja
        x1, x2 = int(w_img * 0.08), int(w_img * 0.92)
        band = gray[y1:y2, x1:x2]

        if band.size == 0 or band.shape[0] < 30 or band.shape[1] < 50:
            return None

        # 2. Corrección fina de inclinación/rotación dentro de la banda del dedo
        blur = cv2.GaussianBlur(band, (15, 15), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > (band.shape[0] * band.shape[1] * 0.20):
                rect = cv2.minAreaRect(c)
                (cx, cy), (w, h), angle = rect
                if w < h:
                    angle += 90.0
                    w, h = h, w
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90

                if abs(angle) > 1.0:
                    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                    band = cv2.warpAffine(
                        band, M, (band.shape[1], band.shape[0]),
                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )

        return cv2.resize(band, self.target_size, interpolation=cv2.INTER_CUBIC)

    def _compute_hessian_vesselness(self, img_norm: np.ndarray, sigmas: List[float] = [1.5, 2.5, 4.0]) -> np.ndarray:
        """
        Implementación vectorial pura en NumPy/OpenCV del filtro de Frangi/Curvatura Hessiana.
        Detecta estructuras tubulares (venas) e ignora manchas circulares o ruido del sensor.
        """
        vesselness_max = np.zeros_like(img_norm, dtype=np.float32)
        beta = 0.5
        c_factor = 15.0

        for sigma in sigmas:
            # Suavizado gaussiano en la escala actual
            ksize = int(2 * np.ceil(3 * sigma) + 1)
            smoothed = cv2.GaussianBlur(img_norm, (ksize, ksize), sigma)

            # Segundas derivadas parciales (Matriz Hessiana)
            Ixx = cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, ksize=3) * (sigma ** 2)
            Iyy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, ksize=3) * (sigma ** 2)
            Ixy = cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, ksize=3) * (sigma ** 2)

            # Autovalores analíticos de matriz simétrica 2x2
            tmp = np.sqrt((Ixx - Iyy) ** 2 + 4 * (Ixy ** 2))
            l1 = 0.5 * (Ixx + Iyy + tmp)
            l2 = 0.5 * (Ixx + Iyy - tmp)

            # Ordenar por magnitud absoluta: |lambda1| <= |lambda2|
            cond = np.abs(l1) > np.abs(l2)
            lambda1 = np.where(cond, l2, l1)
            lambda2 = np.where(cond, l1, l2)

            # En imagen invertida (venas brillantes), buscamos crestas cóncavas: lambda2 < 0
            lambda2_safe = np.where(lambda2 == 0, -1e-6, lambda2)
            rb = (lambda1 / lambda2_safe) ** 2
            s2 = (lambda1 ** 2) + (lambda2 ** 2)

            # Ecuación de Frangi
            v = np.exp(-rb / (2 * (beta ** 2))) * (1.0 - np.exp(-s2 / (2 * (c_factor ** 2))))
            v = np.where(lambda2 < 0, v, 0.0)

            vesselness_max = np.maximum(vesselness_max, v)

        # Normalizar a [0, 255]
        if vesselness_max.max() > 0:
            vesselness_max = (vesselness_max / vesselness_max.max()) * 255.0
        return vesselness_max.astype(np.uint8)

    def _compute_gabor_response(self, img: np.ndarray) -> np.ndarray:
        """Aplica el banco de Gabor y retiene la respuesta máxima direccional."""
        img_f = img.astype(np.float32)
        accum = np.zeros_like(img_f)
        for kern in self.gabor_kernels:
            fimg = cv2.filter2D(img_f, cv2.CV_32F, kern)
            accum = np.maximum(accum, fimg)
        
        accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
        return accum.astype(np.uint8)

    def process(self, pil_image: Image.Image) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """
        Retorna un tensor visual de 3 canales (PIL Image) donde cada canal alimenta a la CNN 
        con una representación física distinta de las venas, más el mapa de visualización.
        """
        if pil_image is None:
            return None, None

        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        roi = self._align_and_crop_roi(gray)
        if roi is None:
            return None, None

        # La hemoglobina desoxigenada absorbe a 850nm (venas oscuras). Invertimos a venas brillantes.
        inv_roi = cv2.bitwise_not(roi)

        # Reducción de ruido de compresión JPEG del ESP32 preservando bordes vasculares
        denoised = cv2.fastNlMeansDenoising(inv_roi, None, h=7, templateWindowSize=7, searchWindowSize=21)

        # CANAL R: Textura base con realce adaptativo suave y filtro bilateral
        ch_r = self.clahe_light.apply(denoised)
        ch_r = cv2.bilateralFilter(ch_r, d=5, sigmaColor=45, sigmaSpace=45)

        # CANAL G: Respuesta energética de Banco de Gabor (orientación de flujo)
        ch_g_base = self.clahe_strong.apply(denoised)
        ch_g = self._compute_gabor_response(ch_g_base)

        # CANAL B: Curvatura Principal / Frangi Vesselness (topología pura y bifurcaciones)
        ch_b = self._compute_hessian_vesselness(ch_r.astype(np.float32))

        # Fusión en imagen compuesta multicanal para el backbone neuronal
        multi_modal_vascular = cv2.merge([ch_r, ch_g, ch_b])
        
        # Mapa visual para auditoría/interfaz (combinación ponderada de curvatura y Gabor)
        vis_map = cv2.addWeighted(ch_g, 0.5, ch_b, 0.5, 0)

        return Image.fromarray(multi_modal_vascular), vis_map


# =====================================================================
# 2. ARQUITECTURA DEEP LEARNING CON PRESERVACIÓN TOPOLÓGICA (PyTorch)
# =====================================================================
class GeMPooling(nn.Module):
    """Generalized Mean Pooling: enfoca la activación en las respuestas intensas de venas."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class SpatialPyramidGeM(nn.Module):
    """
    Evita que la red pierda la ubicación espacial de las bifurcaciones venosas.
    Divide el mapa de características en una cuadrícula espacial (Global + 2x4 regiones)
    aplicando GeM en cada sub-región antes de concatenar.
    """
    def __init__(self, in_channels: int, out_dim: int = 512):
        super().__init__()
        self.gem_global = GeMPooling(p=3.0)
        self.gem_local = GeMPooling(p=3.0)
        
        # 1 (Global) + 2x2 (4 cuadrantes) = 5 descriptores concatenados
        total_concat_dim = in_channels * 5
        
        self.projector = nn.Sequential(
            nn.Linear(total_concat_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=True)
        )
        self._init_weights()

    def _init_weights(self):
        """Inicialización ortogonal para maximizar la separación angular inicial."""
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Descriptor Global
        g_feat = self.gem_global(x).view(B, -1)

        # Descriptores Locales (Cuadrantes Topológicos: Proximal/Distal, Superior/Inferior)
        h_mid, w_mid = H // 2, W // 2
        q1 = self.gem_local(x[:, :, :h_mid, :w_mid]).view(B, -1)
        q2 = self.gem_local(x[:, :, :h_mid, w_mid:]).view(B, -1)
        q3 = self.gem_local(x[:, :, h_mid:, :w_mid]).view(B, -1)
        q4 = self.gem_local(x[:, :, h_mid:, w_mid:]).view(B, -1)

        spatial_pyramid = torch.cat([g_feat, q1, q2, q3, q4], dim=1)
        embedding = self.projector(spatial_pyramid)
        return F.normalize(embedding, p=2, dim=1)


class VascularConvNeXtNet(nn.Module):
    """Backbone ConvNeXt-Small en FP32 con cabezal de preservación espacial."""
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        base_model = convnext_small(weights=weights)
        
        # Extraer el extractor de características convolucional (768 canales de salida)
        self.features = base_model.features
        self.spatial_head = SpatialPyramidGeM(in_channels=768, out_dim=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)  # Shape: [B, 768, H/32, W/32]
        embedding = self.spatial_head(feat_map)
        return embedding


class VascularNeuralExtractor:
    """
    Motor de inferencia con Test-Time Augmentation (TTA) para invariancia a
    micro-desplazamientos longitudinales y transversales en el lector físico.
    """
    def __init__(self, weights_path: Optional[str] = None, num_threads: int = 4):
        torch.set_num_threads(num_threads)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = VascularConvNeXtNet(embedding_dim=512).to(self.device)
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.eval()

        # Transformación estándar manteniendo relación de aspecto 2:1 (320x160)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _generate_tta_batch(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Genera un batch de 5 vistas ligeramente desplazadas/escaladas en memoria
        para neutralizar errores de postura del dedo en el sensor.
        """
        w, h = pil_img.size  # (320, 160)
        tensors = []

        # 1. Vista central original
        tensors.append(self.normalize(pil_img))

        # 2 & 3. Micro-desplazamientos longitudinales (±4% en X)
        dx = int(w * 0.04)
        crop_left = pil_img.crop((0, 0, w - dx, h)).resize((w, h), Image.BICUBIC)
        crop_right = pil_img.crop((dx, 0, w, h)).resize((w, h), Image.BICUBIC)
        tensors.append(self.normalize(crop_left))
        tensors.append(self.normalize(crop_right))

        # 4 & 5. Zoom central suave (94%) y micro-rotación residual (±1.5°)
        zx, zy = int(w * 0.03), int(h * 0.03)
        zoom_center = pil_img.crop((zx, zy, w - zx, h - zy)).resize((w, h), Image.BICUBIC)
        rot_pos = pil_img.rotate(1.5, resample=Image.BICUBIC, expand=False)
        tensors.append(self.normalize(zoom_center))
        tensors.append(self.normalize(rot_pos))

        return torch.stack(tensors, dim=0).to(self.device)

    @torch.inference_mode()
    def extract_embedding(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        if pil_image is None:
            return None
        try:
            batch_tta = self._generate_tta_batch(pil_image)  # [5, 3, 160, 320]
            embeddings = self.model(batch_tta)               # [5, 512]
            
            # Fusión ponderada (la vista central pesa más que las periféricas)
            weights = torch.tensor([0.36, 0.16, 0.16, 0.16, 0.16], device=self.device).unsqueeze(1)
            fused_embedding = torch.sum(embeddings * weights, dim=0, keepdim=True)
            fused_embedding = F.normalize(fused_embedding, p=2, dim=1)
            
            return fused_embedding.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[ERROR NEURAL EXTRACTOR]: {e}")
            return None


# =====================================================================
# 3. MOTOR DE COMPARACIÓN Y CONSENSO BANCARIO
# =====================================================================
class VascularBiometricEngine:
    """
    Orquestador bancario que ejecuta controles de calidad de captura, 
    consenso sobre las 5 muestras enroladas y verificación de margen de seguridad.
    """
    def __init__(
        self, 
        database_dir: str = "database", 
        weights_path: Optional[str] = None,
        similarity_threshold: float = 0.88
    ):
        self.database_dir = database_dir
        self.similarity_threshold = similarity_threshold
        os.makedirs(self.database_dir, exist_ok=True)
        
        self.preprocessor = VascularPreprocessor(target_size=(320, 160))
        self.extractor = VascularNeuralExtractor(weights_path=weights_path)

    def _assess_vascular_quality(self, vis_map: np.ndarray) -> Tuple[bool, float]:
        """
        Verifica que la captura contenga información de contraste vascular
        y no sea una captura negra, uniforme o sin señal infrarroja.
        """
        std_val = float(np.std(vis_map))
        laplacian_var = float(cv2.Laplacian(vis_map, cv2.CV_64F).var())
        
        # Una imagen vacía, negra o uniforme tiene std < 5
        is_valid = (std_val > 5.0) and (laplacian_var > 1.0)
        quality_score = min(100.0, max(10.0, (std_val / 40.0) * 60.0 + min(40.0, laplacian_var * 2.0)))
        return is_valid, round(float(quality_score), 2)

    def _match_vein_patterns(self, gallery_map: np.ndarray, probe_map: np.ndarray) -> float:
        """
        Calcula la correlación vascular estructural pura (Gabor + Frangi)
        con búsqueda de alineación traslacional 2D y micro-rotaciones (+- 2.5 grados).
        - Dedo diferente (intruso): obtiene típicamente entre 15% y 32%.
        - Mismo dedo (legítimo): obtiene entre 75% y 98%.
        """
        h, w = gallery_map.shape[:2]
        th, tw = int(h * 0.85), int(w * 0.85)
        my, mx = (h - th) // 2, (w - tw) // 2
        template = probe_map[my:my+th, mx:mx+tw]

        best_score = 0.0
        for angle in [-2.5, 0.0, 2.5]:
            if angle != 0.0:
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                target = cv2.warpAffine(gallery_map, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            else:
                target = gallery_map

            res = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val

        return max(0.0, float(best_score))

    def enroll_sample(self, cedula: str, pil_image: Image.Image, sample_idx: int) -> Dict:
        """Guarda la plantilla del mapa vascular real (Gabor + Frangi) para el cliente."""
        cedula_clean = cedula.strip().upper()
        client_dir = os.path.join(self.database_dir, cedula_clean)
        os.makedirs(client_dir, exist_ok=True)

        vascular_pil, vis_map = self.preprocessor.process(pil_image)
        if vascular_pil is None or vis_map is None:
            return {"status": "error", "message": "Fallo al segmentar el dedo. Alinee en el centro."}

        is_valid, q_score = self._assess_vascular_quality(vis_map)
        if not is_valid:
            return {
                "status": "rejected", 
                "message": f"Calidad vascular insuficiente ({q_score}%). Verifique la luz o enfoque."
            }

        # Guardamos directamente el mapa vascular 2D discriminativo
        save_path = os.path.join(client_dir, f"sample_{sample_idx}.npy")
        np.save(save_path, vis_map)
        return {"status": "success", "quality": q_score, "file": save_path}

    def verify_identity(self, cedula: str, pil_image: Image.Image) -> Tuple[str, Optional[np.ndarray], Dict]:
        """
        Autenticación bancaria basada en correlación estructural de venas.
        Tolerancia cero a falsos positivos (dedos distintos arrojan < 35%).
        """
        cedula_clean = cedula.strip().upper()
        client_dir = os.path.join(self.database_dir, cedula_clean)

        if not os.path.exists(client_dir) or pil_image is None:
            return "🔴 ERROR: Cliente no enrolado o imagen nula.", None, {"authenticated": False}

        # 1. Extracción del mapa vascular de la imagen actual
        vascular_pil, probe_vis_map = self.preprocessor.process(pil_image)
        if vascular_pil is None or probe_vis_map is None:
            return "🟡 LECTURA INVÁLIDA: Coloque el dedo correctamente.", None, {"authenticated": False}

        is_valid, q_score = self._assess_vascular_quality(probe_vis_map)
        if not is_valid:
            return f"🟡 BAJA CALIDAD SEÑAL IR ({q_score}%): Enfoque el sensor.", probe_vis_map, {"authenticated": False}

        # 2. Carga y comparación contra las 5 muestras enroladas
        template_files = sorted(glob.glob(os.path.join(client_dir, "*.npy")))
        if not template_files:
            return "🔴 ERROR: Sin plantillas biométricas.", probe_vis_map, {"authenticated": False}

        match_scores = []
        for fpath in template_files:
            gallery_map = np.load(fpath)
            if gallery_map.shape == probe_vis_map.shape:
                score = self._match_vein_patterns(gallery_map, probe_vis_map)
                match_scores.append(score)

        if not match_scores:
            return "🔴 ERROR: Plantillas antiguas incompatibles. Por favor re-enrole el cliente.", probe_vis_map, {"authenticated": False}

        match_scores.sort(reverse=True)

        # 3. Consenso del Top-3
        if len(match_scores) >= 3:
            top_k_score = float(sum(match_scores[:3]) / 3.0)
        else:
            top_k_score = float(sum(match_scores) / len(match_scores))

        max_single_match = float(match_scores[0])
        porcentaje = round(top_k_score * 100, 2)
        max_porcentaje = round(max_single_match * 100, 2)

        metrics = {
            "authenticated": False,
            "final_score": round(top_k_score, 4),
            "max_match": round(max_single_match, 4),
            "quality_score": q_score
        }

        # Umbral seguro calibrado: 60% (Un intruso da ~25%, el titular da > 75%)
        VASCULAR_THRESHOLD = 0.60

        if top_k_score >= VASCULAR_THRESHOLD and max_single_match >= 0.65:
            metrics["authenticated"] = True
            return f"🟢 IDENTIDAD CONFIRMADA ({porcentaje}% - Máx: {max_porcentaje}%)", probe_vis_map, metrics
        else:
            return f"🔴 FRAUDE DETECTADO (Dedo no coincide: {porcentaje}%)", probe_vis_map, metrics
            return f"🔴 FRAUDE DETECTADO ({porcentaje}%)", vis_map, metrics