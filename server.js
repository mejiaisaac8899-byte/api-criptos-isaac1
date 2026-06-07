const express = require('express');
const cors = require('cors');
const app = express();

// Activamos CORS para permitir conexiones desde cualquier frontend externo
app.use(cors()); 

// Middleware crucial para recibir los bytes puros (JPEG binario) de la cámara
app.use(express.raw({ type: 'image/jpeg', limit: '10mb' }));

// Variables globales en memoria para la última captura y control de flujo
let ultimaFotoVenografo = null;
let contadorFotos = 0;

// ==========================================
// ENDPOINTS DEL VENÓGRAFO (ESP32-CAM)
// ==========================================

// Endpoint para recibir la foto del ESP32 cada 30 segundos
app.post('/upload', (req, res) => {
    if (!req.body || !Buffer.isBuffer(req.body)) {
        console.log(`⚠️ [SERVER] Intento de subida inválido o cuerpo vacío.`);
        return res.status(400).send('No se recibió un archivo binario válido.');
    }
    
    contadorFotos++;
    ultimaFotoVenografo = req.body;
    
    // MUESTRA EN CONSOLA DEL SERVIDOR: Notificación en tiempo real
    const horaActual = new Date().toLocaleTimeString();
    console.log(`📸 [SERVER] Foto #${contadorFotos} recibida con éxito (${ultimaFotoVenografo.length} bytes) a las ${horaActual}`);
    
    res.status(200).send('Imagen guardada exitosamente en Railway');
});

// Endpoint para renderizar la foto directamente en el navegador
app.get('/ver-foto', (req, res) => {
    if (!ultimaFotoVenografo) {
        return res.status(404).send('<h1>Aún no se ha recibido ninguna foto del Venógrafo</h1><p>Asegúrate de que el ESP32 esté encendido y transmitiendo.</p>');
    }
    
    res.set('Content-Type', 'image/jpeg');
    res.set('Cache-Control', 'no-store'); // Evita congelamientos por caché del navegador
    res.send(ultimaFotoVenografo);
});

// ==========================================
// INICIO DEL SERVIDOR
// ==========================================
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Servidor del Venógrafo encendido y escuchando en el puerto ${PORT}`);
});
