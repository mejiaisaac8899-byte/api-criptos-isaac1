#include "esp_camera.h"
#include <WiFi.h>
#include <U8g2lib.h> 
#include <HTTPClient.h> // <-- Librería añadida para las peticiones al servidor

// ===========================
// Forzar el modelo de placa correcto
// ===========================
#define CAMERA_MODEL_AI_THINKER
#include "board_config.h"

// ===========================
// Credenciales WiFi y Servidor
// ===========================
const char *ssid = "Isaac2";
const char *password = "6313mejia";

#include <Preferences.h>

Preferences preferences;
// URL del Servidor Bancario (Se sobreescribe dinámicamente si hay una guardada en NVS Flash)
String serverUrl = "https://opened-reach-swap-educational.trycloudflare.com/upload";

// Opción 2 (Local ultra-rápida): Si la PC y el ESP32 están en el mismo WiFi
// String serverUrl = "http://192.168.10.6:8000/upload";

// ===========================
// Configuración de Pines
// ===========================
#define TRANSISTOR_PIN 2
#define SENSOR_IR_PIN 4 

// Pines Definitivos para la pantalla ST7920
#define LCD_CLK  14 
#define LCD_DATA 13 
#define LCD_CS   12 

U8G2_ST7920_128X64_1_SW_SPI u8g2(U8G2_R0, LCD_CLK, LCD_DATA, LCD_CS, U8X8_PIN_NONE);

// ===========================
// Variables Globales
// ===========================
// Variables volátiles para que ambos núcleos compartan el estado instantáneo
volatile bool estadoInvertido = false;
volatile bool enviandoFoto = false; // Bandera para controlar el aviso visual en LCD
volatile bool ledsFijos = false;    // Modo prueba: mantiene los LEDs encendidos de forma fija

void setLedsFijos(bool encender) {
  ledsFijos = encender;
  if (ledsFijos) {
    digitalWrite(TRANSISTOR_PIN, HIGH);
    Serial.println("💡 [ESP32] LEDs fijos ACTIVADOS para pruebas");
  } else {
    digitalWrite(TRANSISTOR_PIN, LOW);
    Serial.println("💡 [ESP32] LEDs fijos DESACTIVADOS");
  }
}

bool getLedsFijos() {
  return ledsFijos;
}

// ===========================
// MANEJADORES DE TAREAS (FreeRTOS)
// ===========================
TaskHandle_t TareaPantalla;
TaskHandle_t TareaSubida;

void startCameraServer();
void setupLedFlash();

// ===========================
// FUNCIÓN DE LA PANTALLA (Núcleo 0)
// ===========================
void codigoTareaPantalla(void * parameter) {
  for(;;) {
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_ncenB08_tr); 
      u8g2.drawStr(12, 12, "UNEFA Biopago");
      
      u8g2.setCursor(0, 32);
      u8g2.print("IP: ");
      u8g2.print(WiFi.localIP());

      // --- Lógica de avisos en pantalla ---
      if (ledsFijos) {
        u8g2.setCursor(0, 53);
        u8g2.print(">> LEDS FIJOS [ON]");
      } else if (enviandoFoto) {
        // Si se está subiendo, quitamos el cuadro y mostramos el texto
        u8g2.setCursor(0, 53);
        u8g2.print(">> ENVIANDO FOTO...");
      } else {
        // Estado normal: mostramos el estado del sensor IR
        u8g2.setCursor(0, 50);
        u8g2.print("Sensor IR:");

        if (estadoInvertido) {
          // drawBox dibuja un cuadrado sólido (Relleno)
          u8g2.drawBox(70, 40, 20, 20);
        } else {
          // drawFrame dibuja solo el borde del cuadrado (Vacío)
          u8g2.drawFrame(70, 40, 20, 20);
        }
      }

    } while (u8g2.nextPage());

    // Refresco veloz para ver la reacción en tiempo real
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

// ===========================
// TAREA DE ENVÍO DE IMAGEN BASADA EN EVENTOS (Núcleo 1)
// ===========================
void codigoTareaSubida(void * parameter) {
  for(;;) {
    // estadoInvertido == false significa que el sensor está en LOW (Detectó presencia / Cuadro vacío)
    if (!estadoInvertido) {
      
      // 1. ANTIRREBOTE: Esperamos exactamente 1 segundo (1000 ms)
      vTaskDelay(1000 / portTICK_PERIOD_MS);
      
      // Comprobamos si el brazo sigue firmemente ahí después del segundo
      if (!estadoInvertido) {
        Serial.println("[ESP32] Presencia confirmada. Encendiendo Infrarrojos...");
        
        // 2. ENCENDER LEDS INFRARROJOS
        digitalWrite(TRANSISTOR_PIN, HIGH);
        
        // 3. ESPERAR 3 SEGUNDOS para auto-exposición de la cámara
        vTaskDelay(3000 / portTICK_PERIOD_MS);
        
        if(WiFi.status() == WL_CONNECTED) {
          // 4. CAPTURAR FOTO (Se toma MIENTRAS los LEDs siguen encendidos)
          camera_fb_t * fb = esp_camera_fb_get();
          
          // 5. APAGAR LEDS inmediatamente después de capturar la imagen (si no están fijos para prueba)
          if (!ledsFijos) {
            digitalWrite(TRANSISTOR_PIN, LOW);
          }
          
          if(!fb) {
            Serial.println("❌ [ESP32] Error: Fallo al capturar la cámara");
          } else {
            // 6. ENVIAR FOTO AL SERVIDOR
            enviandoFoto = true; 
            Serial.println("[ESP32] >> Enviando mapeo venoso a Railway...");

            HTTPClient http;
            http.begin(serverUrl);
            http.addHeader("Content-Type", "image/jpeg");
            http.addHeader("X-ESP32-IP", WiFi.localIP().toString()); // Auto-identificación de IP para el backend
            http.addHeader("bypass-tunnel-reminder", "true");
            http.addHeader("ngrok-skip-browser-warning", "true");

            int httpResponseCode = http.POST(fb->buf, fb->len);

            if (httpResponseCode > 0) {
              Serial.printf("✅ [ESP32] Foto recibida en servidor. Código HTTP: %d\n", httpResponseCode);
              String responsePayload = http.getString();
              if (responsePayload.length() > 0 && responsePayload != "OK") {
                Serial.printf("📥 [ESP32] Configuración recibida del servidor: %s\n", responsePayload.c_str());
                sensor_t *s = esp_camera_sensor_get();
                if (s != NULL) {
                  int fIdx = responsePayload.indexOf("\"framesize\":");
                  if (fIdx >= 0) {
                    int val = responsePayload.substring(fIdx + 12).toInt();
                    s->set_framesize(s, (framesize_t)val);
                    Serial.printf("🔧 [ESP32] Resolución (framesize) actualizada a: %d\n", val);
                  }
                  int cIdx = responsePayload.indexOf("\"contrast\":");
                  if (cIdx >= 0) {
                    int val = responsePayload.substring(cIdx + 11).toInt();
                    s->set_contrast(s, val);
                    Serial.printf("🔧 [ESP32] Contraste actualizado a: %d\n", val);
                  }
                  int bIdx = responsePayload.indexOf("\"brightness\":");
                  if (bIdx >= 0) {
                    int val = responsePayload.substring(bIdx + 13).toInt();
                    s->set_brightness(s, val);
                    Serial.printf("🔧 [ESP32] Brillo actualizado a: %d\n", val);
                  }
                  int qIdx = responsePayload.indexOf("\"quality\":");
                  if (qIdx >= 0) {
                    int val = responsePayload.substring(qIdx + 10).toInt();
                    s->set_quality(s, val);
                    Serial.printf("🔧 [ESP32] Calidad JPEG actualizada a: %d\n", val);
                  }
                  int eIdx = responsePayload.indexOf("\"special_effect\":");
                  if (eIdx >= 0) {
                    int val = responsePayload.substring(eIdx + 17).toInt();
                    s->set_special_effect(s, val);
                    Serial.printf("🔧 [ESP32] Efecto actualizado a: %d\n", val);
                  }
                }
              }
            } else {
              Serial.printf("❌ [ESP32] Error de conexión: %s\n", http.errorToString(httpResponseCode).c_str());
            }

            http.end();
            esp_camera_fb_return(fb);
            enviandoFoto = false; 
          }
        } else {
          // Si se cae el WiFi, igual apagamos los LEDs por seguridad si no están fijos
          if (!ledsFijos) {
            digitalWrite(TRANSISTOR_PIN, LOW);
          }
          Serial.println("⚠️ [ESP32] Sin WiFi. Cancelando envío.");
        }
        
        // 7. TIEMPO DE ENFRIAMIENTO (Cooldown)
        Serial.println("[ESP32] Enfriamiento de 5 segundos activado...");
        vTaskDelay(5000 / portTICK_PERIOD_MS);
      }
    }
    
    // Pequeña pausa para no saturar el procesador mientras vigila el sensor
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

// ===========================
// CONFIGURACIÓN PRINCIPAL
// ===========================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // Inicializar almacenamiento no volátil (NVS Flash) para recordar la URL sin cables
  preferences.begin("biopago", false);
  String savedUrl = preferences.getString("serverUrl", "");
  if (savedUrl.length() > 5) {
    serverUrl = savedUrl;
  }
  Serial.printf("[NVS] URL de Servidor activa: %s\n", serverUrl.c_str());

  pinMode(TRANSISTOR_PIN, OUTPUT);
  digitalWrite(TRANSISTOR_PIN, LOW);
  // Mantenemos el INPUT_PULLUP por si la interferencia sigue presente
  pinMode(SENSOR_IR_PIN, INPUT_PULLUP);

  u8g2.begin();
  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(10, 30, "Iniciando Sistema...");
  } while (u8g2.nextPage());

  // Creamos la tarea de pantalla en el Núcleo 0
  xTaskCreatePinnedToCore(
    codigoTareaPantalla,   
    "TareaPantalla",       
    10000,                 
    NULL,                  
    1,                     
    &TareaPantalla,        
    0);                    

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      config.frame_size = FRAMESIZE_SVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    config.frame_size = FRAMESIZE_240X240;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_SVGA); // Predeterminado 800x600 para Biometría Vascular
  }

  WiFi.begin(ssid, password);
  WiFi.setSleep(false);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  // Inicia el servidor local nativo del ESP32 (Para controles y stream web original)
  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");

  // Lanzamos la nueva tarea de subida a la nube en el Núcleo 1
  xTaskCreatePinnedToCore(
    codigoTareaSubida,     // Función
    "TareaSubida",         // Nombre
    8192,                  // Memoria
    NULL,                  // Parámetros
    1,                     // Prioridad
    &TareaSubida,          // Handle
    1                      // Núcleo
  );
}

// ===========================
// LOOP PRINCIPAL (Núcleo 1)
// ===========================
void loop() {
  // Aquí ocurre la lectura del sensor en tiempo real sin bloquear nada
  int estadoActualIR = digitalRead(SENSOR_IR_PIN);
  
  // Lógica invertida: HIGH=Lejos(1.5V) -> true, LOW=Cerca(0V) -> false
  if (estadoActualIR == HIGH) {
    estadoInvertido = true;
  } else {
    estadoInvertido = false;
  }

  delay(10); 
}