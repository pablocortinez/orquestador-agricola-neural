# Guía de Estudio: Fundamentos de Arquitectura de Modelos en el Orquestador Agrícola

## 1. Tensores

### Definición Técnica e Intuición
**Definición Técnica:** Un tensor es una estructura de datos matemática generalizada que representa matrices multidimensionales. En Deep Learning, son el bloque de construcción fundamental para almacenar y procesar datos (entradas, pesos, gradientes). Un escalar es un tensor de dimensión 0 (0D); un vector es 1D; una matriz es 2D; y estructuras de mayor orden son tensores *N*-dimensionales, cada uno caracterizado por su forma (*shape*) y tipo de dato (*dtype*).
**Intuición:** Imagina una hoja de cálculo de Excel: una sola celda es un escalar (0D), una columna es un vector (1D), y la tabla completa es una matriz (2D). Si tienes un libro con múltiples hojas de cálculo por cada mes del año, tienes un tensor 3D. Si además tienes una estantería llena de esos libros (uno por cada sucursal de una empresa), tienes un tensor 4D.

### Relación con la Tarea
En nuestro ecosistema del orquestador agrícola, los tensores son el **medio de transporte universal** de la información visual de las plantas (imágenes sintéticas) y de los parámetros entrenables de nuestra red neuronal. Todo el flujo de datos —desde que se genera la imagen de la hoja enferma hasta que la red emite las probabilidades de patología al agente LangChain— se realiza mediante transformaciones algebraicas sobre tensores utilizando la biblioteca NumPy.

### Mapeo y Fragmentos de Código
En el script, los tensores se inicializan y transforman constantemente para adaptar el formato de los datos a los requisitos de la red neuronal. El hito clave es la creación del conjunto de imágenes y su posterior aplanamiento (*flattening*).

```python
def build_dataset(samples_per_class: int = 220):
    images, labels = [], []
    for class_id in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            images.append(make_leaf_image(class_id))
            labels.append(class_id)
    # 1. Creación del Tensor N-dimensional inicial (NumPy Array)
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

def main():
    images, labels = build_dataset()
    # 2. Transformación del Tensor (Flattening)
    x = images.reshape(len(images), -1)
```

**Análisis de Flujo y Dimensiones (*Shapes*):**
*   **Variable `images` (Tensor Original):** Su dimensión inicial al salir de `build_dataset` es `(660, 32, 32, 3)`.
    *   `660`: Tamaño total del lote o *Batch size* (3 clases x 220 muestras).
    *   `32, 32`: Resolución espacial de la imagen (Alto x Ancho en píxeles).
    *   `3`: Canales de profundidad de color (RGB).
*   **Operación `reshape(len(images), -1)`:** Esta función colapsa las dimensiones espaciales y de canal. La dimensión `len(images)` mantiene el batch en `660`. El `-1` le indica a NumPy que calcule automáticamente el tamaño de la dimensión restante: $32 \times 32 \times 3 = 3072$.
*   **Variable `x` (Tensor Transformado):** El nuevo tensor adopta la forma `(660, 3072)`. Ahora es una matriz 2D plana, el formato estricto que demanda la red implementada para realizar su multiplicación matricial.

---

## 2. Perceptrón Multicapa (MLP)

### Definición Técnica e Intuición
**Definición Técnica:** El MLP es una arquitectura de red neuronal *feedforward* compuesta por capas de nodos (entrada, ocultas, salida). Su característica principal es ser "totalmente conectada" (*dense layer*), lo que significa que cada neurona en la capa $L$ está conectada a absolutamente todas las neuronas de la capa $L-1$. Cada conexión posee un peso ($W$) y un sesgo ($b$), y las capas ocultas aplican funciones de activación no lineales (como ReLU) para aprender representaciones jerárquicas y fronteras de decisión complejas.
**Intuición:** Piensa en un comité de expertos (neuronas) evaluando el rendimiento agrícola. En la primera fila (capa de entrada) están los recolectores de datos de campo, quienes le pasan **todos** sus reportes a cada uno de los analistas en la segunda fila (capa oculta). Cada analista valora (multiplica por sus *pesos*) la información recibida según su especialidad, añade su intuición base (*sesgo*), y si el resultado final supera su umbral de importancia (*activación ReLU*), le transmite su conclusión a la junta directiva (capa de salida), la cual promedia todas las opiniones para emitir el veredicto final (*Softmax*).

### Relación con la Tarea
El MLP se implementa en la clase `SimpleRNA` y actúa como el **"motor perceptivo"** del orquestador. El agente ReAct de LangChain tiene capacidad de razonamiento lógico, pero "es ciego" a los píxeles; depende del MLP para ingerir la imagen (aplanada a 3072 características) y clasificar si hay `Planta_Sana`, `Tizon_Tardio_Papa` u `Oidio_Vid`. Sin las probabilidades que emite este MLP, el LLM no tendría la variable de estado necesaria para decidir consultar tratamientos o reglas climáticas.

### Mapeo y Fragmentos de Código
La arquitectura del MLP y su operación matemática central (*Forward Pass*) se evidencian en la inicialización de sus tensores de pesos y la propagación de matrices:

```python
class SimpleRNA:
    def __init__(self, input_dim: int, hidden_units: int, num_classes: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Inicialización de Pesos para la Capa Oculta 1 (W1) y Salida (W2)
        self.W1 = rng.normal(0, np.sqrt(2 / input_dim), size=(input_dim, hidden_units)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_units), dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_units), size=(hidden_units, num_classes)).astype(np.float32)
        self.b2 = np.zeros((1, num_classes), dtype=np.float32)

    def forward(self, x_batch, dropout_rate=0.0, training=False):
        # Multiplicación matricial Capa Oculta + Activación no lineal
        z1 = x_batch @ self.W1 + self.b1
        a1 = relu(z1)
        
        # [ ... Lógica de Dropout omitida para brevedad ... ]
        
        # Multiplicación matricial Capa de Salida + Softmax
        logits = a1 @ self.W2 + self.b2
        probs = softmax(logits)
        return z1, a1, mask, probs
```

**Análisis Técnico del Flujo de Matrices (*Shapes*):**

| Variable / Matriz | Función Técnica | Dimensión (*Shape*) | Interpretación en la Arquitectura |
| :--- | :--- | :--- | :--- |
| `x_batch` | Entrada del Batch | `(Batch, 3072)` | Lote de imágenes aplanadas; 3072 entradas por muestra. |
| `W1` | Pesos Capa Oculta | `(3072, 64)` | 3072 características proyectadas a 64 unidades ocultas. |
| `z1` | Combinación Lineal | `(Batch, 64)` | Transformación afín: `x_batch @ W1 + b1`. |
| `a1` | Activación ReLU | `(Batch, 64)` | Introduce no linealidad apagando valores negativos. |
| `W2` | Pesos Capa de Salida| `(64, 3)` | 64 activaciones ocultas conectadas a las 3 clases finales. |
| `probs` | Salida Probabilística| `(Batch, 3)` | Distribución Softmax: Las 3 probabilidades suman 1.0. |

---

## 3. Red Neuronal Convolucional (CNN)

### Definición Técnica e Intuición
**Definición Técnica:** Una CNN es una arquitectura especializada en el procesamiento de datos con topología de cuadrícula (como imágenes 2D). En contraposición al MLP que conecta todas las neuronas, la CNN utiliza operaciones de **convolución**: desliza matrices de filtros pequeños (kernels, ej. 3x3) sobre la imagen de entrada para calcular productos punto locales. Esto permite extraer mapas de características (*feature maps*) preservando la jerarquía espacial, logrando "invarianza a la traslación" (detecta patrones independientemente de su posición) y reduciendo masivamente el número de parámetros mediante *weight sharing* (compartición de pesos).
**Intuición:** Piensa en una CNN como un inspector agrícola que utiliza una pequeña lupa cuadrada (el kernel) para revisar una hoja palmo a palmo. En lugar de intentar mirar la hoja completa de un solo vistazo (como lo hace el MLP), el inspector desliza su lupa de izquierda a derecha. Si la lupa tiene un filtro diseñado para detectar texturas irregulares y oscuras, el inspector reaccionará positivamente en el cuadrante exacto donde aparezca una mancha de tizón, sin importarle en qué esquina de la hoja se ubique.

### Relación con la Tarea
Aquí yace una de las lecciones arquitectónicas más importantes del script: **Aunque la Tarea 3 implementa un MLP (`SimpleRNA`), el problema subyacente de diagnosticar enfermedades mediante imágenes es el caso de uso perfecto para una CNN.**
Al usar el MLP, el script se ve forzado a aplanar la imagen, destruyendo las relaciones topológicas (los píxeles vecinos dejan de ser vecinos en el tensor 1D). Una CNN, por el contrario, consumiría el tensor 3D original, permitiendo que la red utilice convoluciones para detectar eficientemente las manchas de Tizón u Oidio (añadidas mediante código) basándose en su forma y textura local, mejorando drásticamente la capacidad de generalización en un entorno real.

### Mapeo y Fragmentos de Código (El "Problema" que la CNN resuelve)
En el código actual, podemos identificar con precisión dónde se generan los patrones geométricos localizados (características espaciales) que una CNN explotaría, y cómo el flujo del MLP los destruye inmediatamente.

```python
# 1. Creación de patrones topológicos locales 
def make_leaf_image(class_id: int) -> np.ndarray:
    # ...
    if class_id == 1: # Generando Tizón Tardío
        for _ in range(np.random.randint(3, 7)):
            add_disc( # ¡Esta función añade manchas en coordenadas (X, Y)!
                image,
                np.random.randint(4, IMG_SIZE - 4),
                np.random.randint(4, IMG_SIZE - 4),
                np.random.randint(2, 5),
                (0.10, 0.08, 0.04),
            )
    return np.clip(image, 0, 1)

# 2. La limitación del modelo actual en main()
def main():
    images, labels = build_dataset()
    # Una CNN ingeriría 'images' directamente: Shape (660, 32, 32, 3)
    
    # El MLP obliga a colapsar la topología 2D:
    x = images.reshape(len(images), -1) 
```

**Análisis Arquitectónico Comparado:**
*   **La Limitación del Enfoque Actual (MLP):** La función `add_disc` dibuja geometrías circulares en un plano 2D. Sin embargo, al aplicar `reshape`, dos píxeles que formaban parte del borde superior e inferior de la misma mancha terminan separados por 32 índices de distancia en el vector plano. El MLP se ve forzado a memorizar estas combinaciones lineales desperdiciando parámetros.
*   **La Solución CNN (Hipotética):** Si integráramos una CNN, procesaría directamente el tensor `(Batch, 32, 32, 3)`. Una primera capa convolucional (ej. `Conv2D(filtros=16, kernel_size=(3,3))`) tendría un tensor de pesos minúsculo de forma `(3, 3, 3, 16)`. Al aplicar la convolución, la CNN detectaría matemáticamente los bordes o variaciones de color de las manchas generadas por `add_disc` en cualquier parte de la hoja, logrando extraer el contexto patológico con mayor exactitud y menor esfuerzo computacional.
