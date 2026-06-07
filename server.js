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
// VISOR WEB CON AUTORECARGA (FRONTEND)
// ==========================================
app.get('/', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Panel del Venógrafo</title>
            <style>
                body { 
                    background-color: #121212; 
                    color: white; 
                    text-align: center; 
                    font-family: Arial, sans-serif; 
                    padding-top: 30px; 
                }
                img { 
                    max-width: 800px; 
                    width: 100%;
                    border: 3px solid #444; 
                    border-radius: 10px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                }
                .status { color: #aaa; margin-top: 15px; font-size: 14px; }
            </style>
        </head>
        <body>
            <h2>Visor de Mapeo Venoso en Tiempo Real</h2>
            
            <img id="imagenVenas" src="/ver-foto" alt="Esperando conexión con la ESP32-CAM..." />
            
            <p class="status">Última actualización: <span id="hora">Esperando datos...</span></p>

            <script>
                // Autorecarga de la imagen cada 2 segundos (2000 ms)
                setInterval(() => {
                    const img = document.getElementById('imagenVenas');
                    
                    // El parámetro ?t= con el tiempo evita que el navegador use la memoria caché
                    // y lo obliga a descargar la foto nueva del servidor.
                    img.src = '/ver-foto?t=' + new Date().getTime();
                    
                    // Actualiza el texto de la hora para confirmar visualmente que está corriendo
                    document.getElementById('hora').innerText = new Date().toLocaleTimeString();
                }, 2000);
            </script>
        </body>
        </html>
    `);
});

// ==========================================
// ENDPOINTS DEL VENÓGRAFO (ESP32-CAM)
// ==========================================

// Endpoint para recibir la foto del ESP32
app.post('/upload', (req, res) => {
    if (!req.body || !Buffer.isBuffer(req.body)) {
        console.log('⚠️ [SERVER] Intento de subida inválido o cuerpo vacío.');
        return res.status(400).send('No se recibió un archivo binario válido.');
    }
    
    contadorFotos++;
    ultimaFotoVenografo = req.body;
    
    // MUESTRA EN CONSOLA DEL SERVIDOR: Notificación en tiempo real
    const horaActual = new Date().toLocaleTimeString();
    console.log(\`📸 [SERVER] Foto #\${contadorFotos} recibida con éxito (\${ultimaFotoVenografo.length} bytes) a las \${horaActual}\`);
    
    res.status(200).send('Imagen guardada exitosamente en Railway');
});

// Endpoint para renderizar la foto cruda directamente (usado por el HTML)
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
    console.log(\`🚀 Servidor del Venógrafo encendido y escuchando en el puerto \${PORT}\`);
});
