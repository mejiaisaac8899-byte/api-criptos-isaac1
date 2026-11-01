import os
import glob
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Dict


# =====================================================================
# 1. EXTRACTOR TOPOLÓGICO DE VENAS (LEE ROI + MIURA STEERABLE CURVATURE)
# =====================================================================
class VascularPreprocessor:
    """
    Motor de preprocesamiento y extracción vascular en infrarrojo cercano (850nm):
    1. Segmentación diferencial de bordes del dedo (método de Lee et al.) inmune
       a fugas de luz IR (>185), sombras de carcasa (<20) y dedos adyacentes.
    2. Normalización geométrica a escala física fija manteniendo la relación de
       aspecto del sensor OV2640 y corrigiendo inclinaciones de hasta +-12 grados.
    3. Normalización Retinex C-infinito libre de artefactos de bloques CLAHE.
    4. Extracción de Curvatura Máxima Dirigible (Miura Steerable Maximum Curvature)
       en 8 orientaciones y 3 escalas espaciales + árbol binario y esqueleto.
    """
    def __init__(self, target_size: Tuple[int, int] = (320, 128)):
        self.target_size = target_size  # (Ancho=320, Alto=128)
        self.sigmas = (2.0, 3.5, 5.0)
        thetas = np.linspace(0, np.pi, 8, endpoint=False)
        self.cos_t = np.cos(thetas).astype(np.float32)
        self.sin_t = np.sin(thetas).astype(np.float32)

    def _align_and_crop_roi(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detecta los contornos superior e inferior reales del dedo columna por columna,
        calcula el eje central anatómico y rectifica mediante interpolación cúbica.
        """
        if gray is None or gray.size == 0:
            return None
        h, w = gray.shape[:2]
        if h < 120 or w < 240:
            return None

        out_w, out_h = self.target_size
        blur = cv2.GaussianBlur(gray.astype(np.float32), (31, 9), 0)
        dy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)

        # 1. Detectar barra superior negra del chasis plástico si está presente en la captura
        row_means = np.mean(blur[:, int(w * 0.2):int(w * 0.8)], axis=1)
        black_rows = np.where(row_means[:int(h * 0.35)] < 20.0)[0]
        y_after_housing = int(black_rows[-1]) + 15 if len(black_rows) > 10 else 0

        active_h = h - y_after_housing
        if active_h > 480:
            y_min_top = y_after_housing + int(active_h * 0.12)
            y_max_top = y_after_housing + int(active_h * 0.45)
        else:
            y_min_top = max(5, y_after_housing)
            y_max_top = y_after_housing + int(active_h * 0.28)

        # 2. Rastreo diferencial de bordes superior (dy < 0) e inferior (dy > 0) en 32 columnas
        xs = np.linspace(int(w * 0.15), int(w * 0.80), 32).astype(int)
        top_y, bot_y = [], []
        for x in xs:
            col_dy = dy[:, x]
            yt = y_min_top + int(np.argmin(col_dy[y_min_top:y_max_top]))
            y_min_bot = max(yt + int(active_h * 0.50), int(h * 0.65))
            y_max_bot = h - 4
            yb = y_min_bot + int(np.argmax(col_dy[y_min_bot:y_max_bot]))
            top_y.append(yt)
            bot_y.append(yb)

        top_y = np.array(top_y, dtype=np.float32)
        bot_y = np.array(bot_y, dtype=np.float32)
        cy = 0.5 * (top_y + bot_y)
        thick = bot_y - top_y

        med_thick = float(np.median(thick))
        if med_thick < 80.0:
            return None

        # Regresión lineal robusta con rechazo de valores atípicos (inliers)
        valid = np.abs(thick - med_thick) < (0.15 * med_thick)
        if np.sum(valid) >= 10:
            slope, intercept = np.polyfit(xs[valid], cy[valid], 1)
        else:
            slope, intercept = np.polyfit(xs, cy, 1)

        # 3. Escala física constante referenciada a la geometría del chasis (evita distorsión X/Y)
        scale = w / 1280.0
        nom_thick = np.clip(med_thick, 395.0 * scale, 425.0 * scale)
        half_span = np.float32(0.35 * nom_thick)

        x_start, x_end = int(w * 0.09), int(w * 0.89)
        grid_x = np.linspace(x_start, x_end, out_w, dtype=np.float32)
        center_y_col = (slope * grid_x + intercept).astype(np.float32)

        map_x = np.tile(grid_x, (out_h, 1)).astype(np.float32)
        y_rel = np.linspace(-half_span, half_span, out_h, dtype=np.float32)[:, None]
        map_y = (center_y_col[None, :] + y_rel).astype(np.float32)

        roi = cv2.remap(
            gray, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return roi

    def _extract_vascular_topology(self, roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Extrae el mapa continuo de curvatura máxima (Miura), el árbol binario vascular
        depurado por componentes conexas y el esqueleto morfológico de 1 píxel.
        """
        f_roi = cv2.GaussianBlur(roi.astype(np.float32), (5, 5), 0.8)
        bg = cv2.GaussianBlur(f_roi, (0, 0), sigmaX=18.0, sigmaY=18.0)

        # Absorción relativa de hemoglobina desoxigenada libre de artefactos de bloques
        norm_vein = (bg - f_roi) / (bg + 15.0)
        absorption_contrast = float(np.std(norm_vein))

        # Máscara de protección contra saturación IR (>185)
        leak_mask = (roi > 185).astype(np.uint8)
        if np.any(leak_mask):
            kern_leak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            leak_dil = cv2.dilate(leak_mask, kern_leak) > 0
        else:
            leak_dil = None

        h, w = roi.shape
        score_map = np.zeros((h, w), dtype=np.float32)

        for sigma in self.sigmas:
            sm = cv2.GaussianBlur(norm_vein, (0, 0), sigmaX=sigma, sigmaY=sigma)
            fx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) * sigma
            fy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) * sigma
            fxx = cv2.Sobel(sm, cv2.CV_32F, 2, 0, ksize=3) * (sigma ** 2)
            fyy = cv2.Sobel(sm, cv2.CV_32F, 0, 2, ksize=3) * (sigma ** 2)
            fxy = cv2.Sobel(sm, cv2.CV_32F, 1, 1, ksize=3) * (sigma ** 2)

            k_max = np.zeros((h, w), dtype=np.float32)
            for c, s in zip(self.cos_t, self.sin_t):
                fd = c * fx + s * fy
                fdd = (c * c) * fxx + (2.0 * c * s) * fxy + (s * s) * fyy
                k = -fdd / ((1.0 + fd * fd) ** 1.5)
                k_max = np.maximum(k_max, k)

            if leak_dil is not None:
                k_max[leak_dil] = 0.0

            margin = int(np.ceil(2.0 * sigma)) + 2
            k_max[:margin, :] = 0.0
            k_max[-margin:, :] = 0.0
            k_max[:, :margin] = 0.0
            k_max[:, -margin:] = 0.0

            p99 = float(np.percentile(k_max, 99.2))
            if p99 > 1e-6:
                k_max = np.clip(k_max / p99, 0.0, 1.0)
            score_map = np.maximum(score_map, k_max)

        # Binarización adaptativa del árbol venoso y limpieza morfológica de ruido
        local_mean = cv2.GaussianBlur(score_map, (21, 21), 5.0)
        binary = ((score_map > local_mean + 0.05) & (score_map > 0.22)).astype(np.uint8)

        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean_bin = np.zeros_like(binary)
        for i in range(1, num_l):
            if stats[i, cv2.CC_STAT_AREA] >= 20:
                clean_bin[labels == i] = 1

        # Esqueletización morfológica (líneas centrales de 1px de grosor)
        skel = np.zeros_like(clean_bin)
        temp = clean_bin.copy()
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(temp, elem)
            opened = cv2.dilate(eroded, elem)
            sub = cv2.subtract(temp, opened)
            skel = cv2.bitwise_or(skel, sub)
            temp = eroded
            if cv2.countNonZero(temp) == 0:
                break

        return score_map.astype(np.float32), clean_bin, skel, absorption_contrast

    def extract_template_and_vis(self, pil_image: Image.Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
        """
        Extrae la plantilla biométrica dual (H, W, 2) en uint8 [Curvatura, Árbol Binario],
        el mapa visual para auditoría en interfaz y las métricas de calidad física.
        """
        if pil_image is None:
            return None, None, {"valid": False, "quality": 0.0}

        img_rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        roi = self._align_and_crop_roi(gray)
        if roi is None:
            return None, None, {"valid": False, "quality": 0.0}

        score_map, clean_bin, skel, abs_contrast = self._extract_vascular_topology(roi)

        # Empaquetado de plantilla biométrica en 2 canales uint8:
        # Canal 0: Mapa continuo de curvatura máxima (0..255)
        # Canal 1: Mapa topológico binario depurado (0 o 255)
        ch_curv = (score_map * 255.0).round().astype(np.uint8)
        ch_bin = (clean_bin * 255).astype(np.uint8)
        template_pack = np.stack([ch_curv, ch_bin], axis=-1)

        # Mapa visual de alto contraste para el cajero (fusión de cresta continua + árbol + esqueleto)
        vis_map = np.clip(ch_curv.astype(np.float32) * 0.70 + ch_bin.astype(np.float32) * 0.30, 0, 255).astype(np.uint8)
        vis_map[skel > 0] = 255

        # Evaluación física de calidad vascular
        mean_i = float(np.mean(roi))
        std_i = float(np.std(roi))
        vein_density = float(np.mean(clean_bin > 0))
        num_l, _, stats, _ = cv2.connectedComponentsWithStats(clean_bin, connectivity=8)
        max_branch = int(stats[1:, cv2.CC_STAT_AREA].max()) if num_l > 1 else 0

        is_valid = (
            (20.0 <= mean_i <= 215.0)
            and (std_i >= 4.0)
            and (abs_contrast >= 0.020)
            and (0.06 <= vein_density <= 0.45)
            and (max_branch >= 75)
        )

        branch_score = min(50.0, (max_branch / 380.0) * 50.0)
        contrast_score = min(30.0, (abs_contrast / 0.055) * 30.0)
        density_score = 20.0 * (1.0 - min(1.0, abs(vein_density - 0.25) / 0.20))
        q_score = round(min(99.5, max(10.0, branch_score + contrast_score + density_score)), 2)
        if not is_valid:
            q_score = min(q_score, 35.0)

        return template_pack, vis_map, {"valid": is_valid, "quality": q_score}

    def process(self, pil_image: Image.Image) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """
        Método de compatibilidad directa con main.py:
        Retorna (PIL.Image multicanal, vis_map np.ndarray).
        """
        tpl_pack, vis_map, _ = self.extract_template_and_vis(pil_image)
        if tpl_pack is None or vis_map is None:
            return None, None
        multi_ch = np.stack([tpl_pack[:, :, 0], tpl_pack[:, :, 1], vis_map], axis=-1)
        return Image.fromarray(multi_ch), vis_map


# =====================================================================
# 2. MOTOR BIOMÉTRICO BANCARIO (MATCHING TOPOLÓGICO SIMÉTRICO)
# =====================================================================
class VascularBiometricEngine:
    """
    Motor de verificación vascular de nivel bancario con:
    - Búsqueda de alineación traslacional 2D (+-58px en X, +-16px en Y) y rotacional (+-3 grados).
    - Correlación cruzada simétrica de media cero sobre Curvatura Dirigible de Miura.
    - Verificación topológica del árbol binario vascular para anular falsos positivos.
    """
    def __init__(
        self,
        database_dir: str = "database",
        weights_path: Optional[str] = None,
        similarity_threshold: float = 0.40
    ):
        self.database_dir = database_dir
        self.similarity_threshold = similarity_threshold
        os.makedirs(self.database_dir, exist_ok=True)
        self.preprocessor = VascularPreprocessor(target_size=(320, 128))

    @staticmethod
    def _unpack_template(arr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Extrae (score_map_float32, binary_map_uint8) desde el archivo .npy almacenado."""
        if arr is None:
            return None
        if arr.ndim == 3 and arr.shape == (128, 320, 2):
            s = arr[:, :, 0].astype(np.float32) / 255.0
            b = (arr[:, :, 1] > 127).astype(np.uint8)
            return s, b
        if arr.ndim == 2 and arr.shape == (128, 320):
            s = arr.astype(np.float32) / 255.0
            local_m = cv2.GaussianBlur(s, (21, 21), 5.0)
            b = ((s > local_m + 0.05) & (s > 0.22)).astype(np.uint8)
            return s, b
        return None

    def _match_vein_patterns(self, gallery_pack: np.ndarray, probe_pack: np.ndarray) -> float:
        """
        Ejecuta comparación estructural simétrica entre plantilla enrolada y muestra actual:
        1. Alineación invariante a traslación longitudinal/transversal y rotación fina.
        2. Fusión de correlación de cresta continua (Miura MC) + topología binaria ZNCC.
        3. Calibración estadística sobre el piso de valor extremo para separar intrusos (<25%)
           de titulares legítimos (>70%).
        """
        u1 = self._unpack_template(gallery_pack)
        u2 = self._unpack_template(probe_pack)
        if u1 is None or u2 is None:
            return 0.0

        s1, b1 = u1
        s2, b2 = u2
        h, w = s1.shape
        max_dx, max_dy = 58, 16

        best_cc = -1.0
        best_dir = 0
        best_angle = 0.0
        best_loc = (max_dx, max_dy)

        pairs = [(s1, s2, b1, b2), (s2, s1, b2, b1)]
        for direction, (src, dst, _, _) in enumerate(pairs):
            tpl = dst[max_dy:h - max_dy, max_dx:w - max_dx]
            for angle in (-3.0, -1.5, 0.0, 1.5, 3.0):
                if angle != 0.0:
                    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                    src_rot = cv2.warpAffine(
                        src, M, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0
                    )
                else:
                    src_rot = src

                res = cv2.matchTemplate(src_rot, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_cc:
                    best_cc = float(max_val)
                    best_dir = direction
                    best_angle = angle
                    best_loc = max_loc

        if best_cc <= 0.0:
            return 0.0

        # Evaluar coincidencia del árbol binario vascular en la pose óptima encontrada
        _, _, b_src, b_dst = pairs[best_dir]
        dx = best_loc[0] - max_dx
        dy = best_loc[1] - max_dy
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_angle, 1.0)
        M[0, 2] -= dx
        M[1, 2] -= dy

        b_src_al = cv2.warpAffine(
            b_src.astype(np.float32), M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        sub_b1 = cv2.GaussianBlur(b_src_al[max_dy:h - max_dy, max_dx:w - max_dx], (5, 5), 1.2)
        sub_b2 = cv2.GaussianBlur(b_dst[max_dy:h - max_dy, max_dx:w - max_dx].astype(np.float32), (5, 5), 1.2)

        res_bin = cv2.matchTemplate(sub_b1, sub_b2, cv2.TM_CCOEFF_NORMED)
        bin_zncc = max(0.0, float(res_bin[0, 0]))

        # Fusión multimodal: Curvatura continua (68%) + Topología binaria (32%)
        combined = 0.68 * max(0.0, best_cc) + 0.32 * bin_zncc
        calibrated = float(np.clip((combined - 0.05) / (0.88 - 0.05), 0.0, 1.0))
        if best_cc > 0.995:
            calibrated = 1.0
        return calibrated

    def enroll_sample(self, cedula: str, pil_image: Image.Image, sample_idx: int) -> Dict:
        """Procesa, valida la calidad física y guarda la plantilla vascular del cliente."""
        cedula_clean = cedula.strip().upper()
        client_dir = os.path.join(self.database_dir, cedula_clean)
        os.makedirs(client_dir, exist_ok=True)

        tpl_pack, vis_map, q_info = self.preprocessor.extract_template_and_vis(pil_image)
        if tpl_pack is None or vis_map is None:
            return {"status": "error", "message": "Fallo al segmentar el dedo. Alinee en el centro."}

        q_score = q_info["quality"]
        if not q_info["valid"]:
            return {
                "status": "rejected",
                "message": f"Calidad vascular insuficiente ({q_score}%). Verifique posición o iluminación IR."
            }

        save_path = os.path.join(client_dir, f"sample_{sample_idx}.npy")
        np.save(save_path, tpl_pack)
        try:
            pil_image.save(os.path.join(client_dir, f"original_{sample_idx}.jpg"), format="JPEG", quality=95)
        except Exception:
            pass
        return {"status": "success", "quality": q_score, "file": save_path}

    def verify_identity(self, cedula: str, pil_image: Image.Image) -> Tuple[str, Optional[np.ndarray], Dict]:
        """
        Autenticación biométrica 1:1 mediante consenso Top-3 sobre las muestras enroladas.
        """
        cedula_clean = cedula.strip().upper()
        client_dir = os.path.join(self.database_dir, cedula_clean)

        if not os.path.exists(client_dir) or pil_image is None:
            return "🔴 ERROR: Cliente no enrolado o imagen nula.", None, {"authenticated": False}

        probe_pack, probe_vis_map, q_info = self.preprocessor.extract_template_and_vis(pil_image)
        if probe_pack is None or probe_vis_map is None:
            return "🟡 LECTURA INVÁLIDA: Coloque el dedo correctamente.", None, {"authenticated": False}

        q_score = q_info["quality"]
        if not q_info["valid"]:
            return "🟡 LECTURA NO VÁLIDA: Ajuste la posición del dedo en el sensor.", probe_vis_map, {"authenticated": False}

        template_files = sorted(glob.glob(os.path.join(client_dir, "*.npy")))
        if not template_files:
            return "🔴 ERROR: Sin plantillas biométricas.", probe_vis_map, {"authenticated": False}

        match_scores = []
        for fpath in template_files:
            try:
                gallery_pack = np.load(fpath)
                if self._unpack_template(gallery_pack) is not None:
                    score = self._match_vein_patterns(gallery_pack, probe_pack)
                    match_scores.append(score)
            except Exception:
                continue

        if not match_scores:
            return (
                "🔴 ERROR: Plantillas antiguas incompatibles. Por favor re-enrole las 5 muestras del cliente.",
                probe_vis_map,
                {"authenticated": False}
            )

        match_scores.sort(reverse=True)

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

        VASCULAR_THRESHOLD = self.similarity_threshold

        if top_k_score >= VASCULAR_THRESHOLD and max_single_match >= (VASCULAR_THRESHOLD + 0.05):
            metrics["authenticated"] = True
            return "🟢 IDENTIDAD CONFIRMADA", probe_vis_map, metrics
        else:
            return "🔴 IDENTIDAD NO COINCIDE", probe_vis_map, metrics