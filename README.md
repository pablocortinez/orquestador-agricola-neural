# 🌱 Orquestador Agrícola Neural (Edge AI)

Este proyecto es una arquitectura de grado profesional diseñada para el diagnóstico de patologías agrícolas y la recomendación de tratamientos en tiempo real. Combina **Deep Learning (PyTorch)**, **MLOps (FastAPI)** y **Orquestación Agéntica (n8n + Gemini)** operando en un entorno local (Edge Computing) para garantizar mínima latencia y máxima fiabilidad.

Este sistema es el resultado de la evolución de un prototipo académico hacia una solución de ingeniería de software robusta, ideal para proyectos de título y despliegues en el mundo real.

---

## 🏗️ Arquitectura del Sistema

El proyecto opera bajo una topología de tres capas fuertemente desacopladas:

1. **Capa de Visión por Computador (Deep Learning)**
   - Red Neuronal Convolucional (CNN) personalizada (`AgricolaCNN`) entrenada con el dataset *PlantVillage*.
   - Clasifica hojas en tres categorías críticas: `Planta Sana`, `Tizón Tardío (Papa)` y `Oídio (Vid)`.
   - El modelo entrenado está persistido en `modelo_vision.pth`.

2. **Capa de MLOps y Microservicios (Backend)**
   - API REST ultrarrápida construida con **FastAPI** (`src/api_vision.py`).
   - Implementa un patrón de *Lifespan Management* para precargar los tensores del modelo directamente en la memoria RAM al arrancar el servidor, reduciendo el tiempo de inferencia a escasos milisegundos.

3. **Capa de Orquestación Agéntica y Reglas de Negocio (n8n)**
   - Flujo visual y automatizado administrado por **n8n**.
   - Integra la inferencia visual local con contexto ambiental en tiempo real (consultando a la API de **OpenWeatherMap**).
   - Utiliza **Google Gemini (Flash)** como Agente Experto, combinando el diagnóstico clínico y el factor climático (ej. no aplicar pesticidas si hay pronóstico de lluvia) para emitir una orden de tratamiento estructurada.

---

## 📂 Estructura del Repositorio

```text
📁 orquestador-agricola-neural/
├── 📁 data/                          # Dataset local — NO versionado en git (ver sección Dataset)
├── 📁 docs/                          # Documentación teórica y guías de estudio del modelo.
├── 📁 src/                           # Código fuente de producción.
│   ├── api_vision.py                 # Microservicio FastAPI para inferencia del modelo.
│   └── entrenar_cnn.py               # Script de entrenamiento PyTorch (exporta .pth).
├── modelo_vision.pth                 # Pesos de la red neuronal entrenada con datos reales.
├── n8n_workflow_demo.json            # Flujo de n8n para pruebas interactivas vía Formulario Web.
├── n8n_workflow_telegram.json        # Flujo activo: bot @FitoScanAIBot (Telegram → CNN → Clima → Gemini).
├── n8n_workflow_final.json           # Flujo de n8n para despliegue automatizado vía Webhooks.
├── mapa_flujo_telegram_agricola.html # Mockup interactivo del flujo objetivo (bot de Telegram).
├── SESION.md                         # Bitácora del proyecto: configuración, historial y decisiones.
└── README.md                         # Este documento.
```

---

## 🚀 Guía de Instalación y Ejecución

Al ser una arquitectura de procesamiento local (*Edge*), la instalación requiere correr el backend de Python y el orquestador en la misma máquina.

### 1. Instalar dependencias Python (solo la primera vez)

```bash
pip install fastapi uvicorn torch torchvision pillow python-multipart
```

### 2. Levantar el Motor de Inferencia (FastAPI)
Abre una terminal en la raíz del proyecto y arranca el servidor. Esto cargará el modelo `.pth` en memoria.

```bash
# En Windows
uvicorn src.api_vision:app --port 8001
```
*Verás un mensaje verde indicando: `✅ Modelo PyTorch cargado en memoria desde 'modelo_vision.pth'`.*

### 3. Levantar el Orquestador (n8n local)
Abre **una segunda terminal** e inicia n8n usando Node.js:

```bash
npx n8n
```
Esto levantará el servidor de n8n en `http://localhost:5678`.

### 4. Configurar el Flujo de Datos
1. Entra a `http://localhost:5678` en tu navegador web.
2. Crea un nuevo Workflow y haz clic en **Import from File...**
3. Selecciona `n8n_workflow_demo.json`.
4. Agrega tus credenciales en los nodos correspondientes:
   - **Nodo API Clima**: Reemplaza el parámetro `appid` con tu API Key de OpenWeatherMap.
   - **Nodo Agrónomo (Gemini)**: Conecta tus credenciales de Google AI (Gemini).

---

## 🧪 Cómo probar el Sistema (Demo)

El archivo `n8n_workflow_demo.json` incluye un **Form Trigger**, lo cual genera una pequeña interfaz web amigable para demostraciones de presentación de título.

1. Dentro de n8n, entra al primer nodo (*Subir Imagen UI*).
2. Haz clic en el botón naranja **"Test step"** y abre la **Test URL** que aparecerá.
3. Desde tu navegador, sube cualquier imagen de la carpeta `data/Tizon_Tardio_Papa/` u `Oidio_Vid/`.
4. Vuelve a n8n y observa cómo la información viaja por los nodos:
   - El nodo **FastAPI** reconoce la enfermedad en tiempo real.
   - El nodo **OpenWeatherMap** captura el clima actual de la zona.
   - El nodo **Gemini** cruza la información clínica + meteorológica para darte el veredicto final.

---

## 📊 Dataset

El dataset **no está versionado en git** por su tamaño (~18.000 imágenes). Descárgalo desde Kaggle y organiza la carpeta `data/` con la siguiente estructura:

**Fuente:** [kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

```text
data/
├── Oidio_Vid/                  # 1000 imágenes  — Oídio en vid
├── Planta_Sana/                # ~1000 imágenes — Hojas sanas (papa + tomate + pimiento)
├── Tizon_Tardio_Papa/          # 1000 imágenes  — Tizón tardío (Phytophthora infestans)
├── Tizon_Temprano_Papa/        # 1000 imágenes  — Tizón temprano (Alternaria solani)
├── Mancha_Bact_Pimiento/       #  997 imágenes  — Mancha bacteriana en pimiento
├── Mancha_Bact_Tomate/         # 2127 imágenes  — Mancha bacteriana en tomate
├── Tizon_Temprano_Tomate/      # 1000 imágenes  — Tizón temprano en tomate
├── Tizon_Tardio_Tomate/        # 1909 imágenes  — Tizón tardío en tomate
├── Moho_Foliar_Tomate/         #  952 imágenes  — Moho foliar (Fulvia fulva)
├── Septoria_Tomate/            # 1771 imágenes  — Septoriosis foliar
├── Arana_Roja_Tomate/          # 1676 imágenes  — Araña roja (Tetranychus urticae)
├── Mancha_Diana_Tomate/        # 1404 imágenes  — Mancha diana (Corynespora)
├── Virus_Rizo_Tomate/          # 3209 imágenes  — Virus del rizado amarillo (TYLCV)
└── Mosaico_Tomate/             #  373 imágenes  — Virus del mosaico del tomate (ToMV)
```

**Mapeo desde Kaggle → carpetas del proyecto:**

| Carpeta Kaggle | Carpeta `data/` |
|---|---|
| `Potato___Late_blight` | `Tizon_Tardio_Papa` |
| `Potato___Early_blight` | `Tizon_Temprano_Papa` |
| `Potato___healthy` + `Tomato_healthy` + `Pepper__bell___healthy` | `Planta_Sana` (cap. ~1000) |
| `Tomato_Bacterial_spot` | `Mancha_Bact_Tomate` |
| `Tomato_Early_blight` | `Tizon_Temprano_Tomate` |
| `Tomato_Late_blight` | `Tizon_Tardio_Tomate` |
| `Tomato_Leaf_Mold` | `Moho_Foliar_Tomate` |
| `Tomato_Septoria_leaf_spot` | `Septoria_Tomate` |
| `Tomato_Spider_mites_Two_spotted_spider_mite` | `Arana_Roja_Tomate` |
| `Tomato__Target_Spot` | `Mancha_Diana_Tomate` |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | `Virus_Rizo_Tomate` |
| `Tomato__Tomato_mosaic_virus` | `Mosaico_Tomate` |
| `Pepper__bell___Bacterial_spot` | `Mancha_Bact_Pimiento` |
| *(fuente externa)* | `Oidio_Vid` |

---

## 🛠️ Escalabilidad Futura (Roadmap)
- **Bot de Telegram** ✅ *(implementado)*: `@FitoScanAIBot` recibe fotos desde el celular del agricultor vía Telegram. Flujo documentado en `n8n_workflow_telegram.json` con 3 rutas de respuesta según la confianza del diagnóstico (≥0.80 diagnóstico definitivo, 0.60-0.79 tentativo, <0.60 pide nueva foto).
- **Despliegue Cloud**: Migrar el flujo n8n a una instancia cloud para que funcione sin depender del PC local. Flujo de producción disponible en `n8n_workflow_final.json`.
- **Entrenamiento Continuo**: Configurar un *pipeline* que reentrene el modelo periódicamente agregando nuevas clasificaciones de plagas a `data/`.

---
*Desarrollado como Proyecto de Título aplicando estándares modernos de Data Engineering e Inteligencia Artificial.*
