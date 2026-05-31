const express = require('express');
const cors = require('cors'); // <-- 1. Importamos CORS
const app = express();

// <-- 2. Activamos CORS para que tu HTML pueda leer la API sin ser bloqueado
app.use(cors()); 

const PORT = process.env.PORT || 3000;

const criptosSoportadas = {
    'bitcoin': 'BTCUSDT',
    'ethereum': 'ETHUSDT',
    'usdt': 'USDCUSDT' 
};

app.get('/api/precio/:moneda', async (req, res) => {
    const monedaRequerida = req.params.moneda.toLowerCase();

    if (!criptosSoportadas[monedaRequerida]) {
        return res.status(404).json({ error: "Moneda no soportada." });
    }

    const simbolo = criptosSoportadas[monedaRequerida];

    try {
        // <-- 3. Usamos MEXC (Mismo formato que Binance, pero sin bloqueos de IP)
        const respuesta = await fetch(`https://api.mexc.com/api/v3/ticker/price?symbol=${simbolo}`);
        const datos = await respuesta.json();

        res.status(200).json({
            criptomoneda: monedaRequerida,
            precio_actual: datos.price
        });

    } catch (error) {
        console.log(`[ERROR] Falló la petición: ${error.message}`);
        res.status(500).json({ error: "Error al consultar el precio." });
    }
});

app.listen(PORT, () => {
    console.log(`¡Backend encendido en el puerto ${PORT}!`);
});
