const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;

const criptosSoportadas = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'usdt': 'USDCUSDT' 
};

app.get('/api/precio/:moneda', async (req, res) => {
    const monedaRequerida = req.params.moneda.toLowerCase();
    
    // --- EVENTO 1: Llega una petición ---
    console.log(`[ALERTA] El ESP32 acaba de pedir el precio de: ${monedaRequerida}`);

    if (!criptosSoportadas[monedaRequerida]) {
        // --- EVENTO 2: Error de petición ---
        console.log(`[ERROR] La moneda ${monedaRequerida} no está en nuestro diccionario. Abortando.`);
        return res.status(404).json({ error: "Moneda no soportada." });
    }

    const simboloBinance = criptosSoportadas[monedaRequerida];

    try {
        // --- EVENTO 3: Comunicándose con Binance ---
        console.log(`[INFO] Consultando a Binance el símbolo: ${simboloBinance}...`);
        
        const respuestaBinance = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${simboloBinance}`);
        const datosBinance = await respuestaBinance.json();

        // --- EVENTO 4: Datos recibidos y listos para enviar ---
        console.log(`[ÉXITO] Binance respondió: ${datosBinance.price}. Enviando respuesta al ESP32.`);

        res.status(200).json({
            criptomoneda: monedaRequerida,
            precio_actual: datosBinance.price
        });

    } catch (error) {
        console.log(`[FALLO CRÍTICO] Ocurrió un error al hablar con Binance: ${error.message}`);
        res.status(500).json({ error: "Error al comunicarse con Binance" });
    }
});

app.listen(PORT, () => {
    console.log(`¡Backend encendido! Escuchando peticiones en el puerto ${PORT}...`);
});
