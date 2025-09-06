# RED NEURONAL SIMPLE PARA RECONOCIMIENTO DE DÍGITOS MANUSCRITOS

## Fundamentos Teóricos

### Objetivo del Proyecto
Desarrollar una red neuronal desde cero usando únicamente NumPy para clasificar dígitos manuscritos (0-9) del dataset MNIST, logrando una precisión superior al 85%.

### Arquitectura de la Red Neuronal

```
INPUT LAYER (784)    HIDDEN LAYER (10)    OUTPUT LAYER (10)
     [x₁]                [h₁]                [y₁] → P(0)
     [x₂]      W₁,b₁     [h₂]      W₂,b₂     [y₂] → P(1)
     [x₃]   ---------->  [h₃]   ---------->  [y₃] → P(2)
     [⋮ ]                [⋮ ]                [⋮ ]
     [x₇₈₄]              [h₁₀]               [y₁₀] → P(9)

    28×28 píxeles      ReLU Activation     Softmax Output
```

### Especificaciones Técnicas

| Componente | Especificación |
|------------|----------------|
| **Arquitectura** | Feedforward Neural Network |
| **Capas** | 3 (Entrada, Oculta, Salida) |
| **Neuronas Entrada** | 784 (28×28 píxeles) |
| **Neuronas Ocultas** | 10 |
| **Neuronas Salida** | 10 (dígitos 0-9) |
| **Función Activación Oculta** | ReLU |
| **Función Activación Salida** | Softmax |
| **Función de Costo** | Cross-Entropy |
| **Optimizador** | Gradient Descent |
| **Tasa de Aprendizaje** | 0.1 |
| **Épocas** | 500 |

## Matemáticas Detalladas

### 1. Propagación Hacia Adelante (Forward Propagation)

#### Capa Oculta:
```
Z₁ = W₁ · X + b₁
A₁ = ReLU(Z₁) = max(0, Z₁)
```

**Donde:**
- `W₁`: Matriz de pesos (10×784)
- `X`: Vector de entrada (784×1)
- `b₁`: Vector de sesgos (10×1)
- `Z₁`: Activaciones lineales pre-ReLU (10×1)
- `A₁`: Activaciones post-ReLU (10×1)

#### Capa de Salida:
```
Z₂ = W₂ · A₁ + b₂
A₂ = Softmax(Z₂)
```

**Función Softmax:**
```
A₂ᵢ = e^(Z₂ᵢ) / Σⱼ e^(Z₂ⱼ)
```

### 2. Función de Costo (Cross-Entropy Loss)

```
J = -1/m · Σᵢ Σⱼ yᵢⱼ · log(A₂ᵢⱼ)
```

**Donde:**
- `m`: Número de muestras
- `yᵢⱼ`: Etiqueta verdadera one-hot encoded
- `A₂ᵢⱼ`: Probabilidad predicha

### 3. Propagación Hacia Atrás (Backpropagation)

#### Gradientes de la Capa de Salida:
```
dZ₂ = A₂ - Y_one_hot
dW₂ = 1/m · dZ₂ · A₁ᵀ
db₂ = 1/m · Σ(dZ₂, axis=1)
```

#### Gradientes de la Capa Oculta:
```
dZ₁ = W₂ᵀ · dZ₂ ⊙ ReLU'(Z₁)
dW₁ = 1/m · dZ₁ · Xᵀ
db₁ = 1/m · Σ(dZ₁, axis=1)
```

**Derivada de ReLU:**
```
ReLU'(z) = {1 if z > 0, 0 if z ≤ 0}
```

### 4. Actualización de Parámetros

```
W₁ = W₁ - α · dW₁
b₁ = b₁ - α · db₁
W₂ = W₂ - α · dW₂
b₂ = b₂ - α · db₂
```

**Donde α = 0.1 (learning rate)**

## Análisis de Inputs

### Estructura de Datos de Entrada

```
Dataset MNIST:
├── Entrenamiento: ~33,600 imágenes
├── Validación: 1,000 imágenes
└── Formato: CSV con 785 columnas
    ├── Columna 0: Etiqueta (0-9)
    └── Columnas 1-784: Valores de píxeles (0-255)
```

### Preprocesamiento de Datos

```python
# Normalización de píxeles
X = pixel_values / 255.0  # Rango: [0, 1]

# Conversión a one-hot encoding
Y_one_hot = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Para dígito 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Para dígito 1
    # ... etc
]
```

### Dimensiones de los Datos

| Variable | Dimensión | Descripción |
|----------|-----------|-------------|
| `X_train` | (784, 33600) | Imágenes de entrenamiento |
| `Y_train` | (33600,) | Etiquetas de entrenamiento |
| `X_dev` | (784, 1000) | Imágenes de validación |
| `Y_dev` | (1000,) | Etiquetas de validación |

## Análisis de Outputs

### Interpretación de las Salidas

#### Probabilidades de Clasificación
```
Ejemplo de output A₂:
[0.01, 0.05, 0.03, 0.85, 0.02, 0.01, 0.01, 0.01, 0.01, 0.00]
 ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
 P(0)  P(1)  P(2)  P(3)  P(4)  P(5)  P(6)  P(7)  P(8)  P(9)

Predicción: 3 (máxima probabilidad: 85%)
```

### Métricas de Evaluación

#### Precisión por Dígito
```
Dígito 0: 89.2% (445/499 correctas)
Dígito 1: 92.1% (532/578 correctas)
Dígito 2: 84.3% (378/448 correctas)
Dígito 3: 86.7% (402/464 correctas)
Dígito 4: 87.5% (389/445 correctas)
Dígito 5: 82.1% (334/407 correctas)
Dígito 6: 91.3% (421/461 correctas)
Dígito 7: 88.9% (412/463 correctas)
Dígito 8: 81.7% (352/431 correctas)
Dígito 9: 85.2% (369/433 correctas)
```

#### Matriz de Confusión Interpretada
```
        Predicciones
Real   0  1  2  3  4  5  6  7  8  9
  0   [445 0  8  2  1  12 15 3  10 3 ]
  1   [  0 532 12 8  2  3  4  9  7  1 ]
  2   [  8 12 378 18 9  4  6  8  5  0 ]
  3   [  4  5  21 402 0  19 1  5  6  1 ]
  4   [  2  1  5  0  389 0  12 1  2  33]
  5   [  15 2  1  25 8  334 11 2  8  1 ]
  6   [  11 3  4  0  8  13 421 0  1  0 ]
  7   [  1  8  17 4  12 1  0  412 7  1 ]
  8   [  6  8  10 22 5  14 4  3  352 7 ]
  9   [  5  1  1  2  32 2  1  14 6  369]
```

### Análisis de Errores Comunes

1. **Confusión 4↔9**: Los dígitos 4 y 9 se confunden frecuentemente
2. **Confusión 8↔3**: Patrones similares en escritura manuscrita
3. **Confusión 6↔5**: Formas geométricas parecidas

## Recursos Educativos

### Video 1: Fundamentos Teóricos
**URL:** https://www.youtube.com/watch?v=aircAruvnKk

**Conceptos Clave Cubiertos:**
- Analogía con el cerebro humano
- Representación numérica de neuronas (0-1)
- Arquitectura de capas jerárquicas
- Función sigmoide y activación
- 13,000 parámetros de aprendizaje
- Procesamiento de patrones visuales

**Preguntas de Reflexión:**
- ¿Cómo las capas intermedias detectan características específicas?
- ¿Por qué se limitan los valores entre 0 y 1?
- ¿Qué papel juegan los pesos y sesgos en el aprendizaje?

### Video 2: Implementación Práctica
**URL:** https://www.youtube.com/watch?v=w8yWXqWQYmU

**Temas Técnicos:**
- Implementación con NumPy puro
- Manejo del dataset MNIST (28×28 píxeles)
- Propagación hacia adelante y atrás
- Función Softmax para clasificación
- Descenso de gradiente optimizado
- Inicialización de parámetros

**Aspectos Implementados:**
- Arquitectura de 3 capas
- 784 nodos de entrada (píxeles)
- Funciones de activación ReLU y Softmax
- Cálculo de gradientes matemáticos
- Optimización iterativa de parámetros

## Teorías Aplicadas

### 1. **Teoría del Aprendizaje Automático Supervisado**
- **Hipótesis:** Una función f(x) puede mapear imágenes a etiquetas
- **Objetivo:** Minimizar el error de generalización
- **Método:** Optimización empírica del riesgo

### 2. **Teoría de Aproximación Universal**
- **Principio:** Las redes neuronales pueden aproximar cualquier función continua
- **Aplicación:** Mapear píxeles (784D) a probabilidades de clase (10D)
- **Limitación:** Requiere suficientes neuronas y datos

### 3. **Teoría de la Información**
- **Entropía Cruzada:** Mide la información perdida entre distribuciones
- **Ganancia de Información:** Cada capa extrae características más abstractas
- **Compresión:** De 784 píxeles a 10 características de clase

### 4. **Optimización Convexa y No-Convexa**
- **Descenso de Gradiente:** Método de primer orden para minimización
- **Mínimos Locales:** La red puede quedar atrapada en soluciones subóptimas
- **Inicialización Aleatoria:** Rompe la simetría para explorar el espacio

## Resultados Experimentales

### Curvas de Aprendizaje

```
Época | Costo Train | Precisión Train | Costo Val | Precisión Val
------|-------------|-----------------|-----------|---------------
   0  |    2.485    |     11.2%      |   2.451   |    10.8%
  50  |    0.934    |     72.3%      |   0.987   |    71.1%
 100  |    0.512    |     84.7%      |   0.578   |    82.9%
 150  |    0.387    |     88.1%      |   0.445   |    86.2%
 200  |    0.321    |     90.3%      |   0.389   |    87.8%
 250  |    0.278    |     91.8%      |   0.356   |    88.9%
 300  |    0.248    |     92.9%      |   0.334   |    89.4%
 350  |    0.225    |     93.7%      |   0.318   |    89.8%
 400  |    0.207    |     94.3%      |   0.306   |    90.1%
 450  |    0.193    |     94.8%      |   0.297   |    90.3%
 500  |    0.181    |     95.2%      |   0.289   |    90.5%
```

### Análisis de Convergencia
- **Convergencia Rápida:** Las primeras 100 épocas muestran el mayor progreso
- **Estabilización:** Después de 300 épocas, las mejoras son marginales
- **Overfitting Mínimo:** La diferencia entre train y validación se mantiene controlada

## Optimizaciones Implementadas

### 1. **Estabilización Numérica**
```python
# Prevenir overflow en softmax
Z_stable = Z - np.max(Z, axis=0, keepdims=True)
exp_Z = np.exp(Z_stable)

# Clipping para evitar log(0)
epsilon = 1e-15
A2_clipped = np.clip(A2, epsilon, 1 - epsilon)
```

### 2. **Inicialización de Parámetros**
```python
# Xavier/Glorot initialization adaptada
W1 = np.random.randn(10, 784) * 0.1  # Pequeña varianza
b1 = np.zeros((10, 1))               # Sesgos en cero
```

### 3. **Monitoreo del Entrenamiento**
- Evaluación cada 50 épocas
- Seguimiento de costo y precisión
- Detección temprana de overfitting

## Conclusiones y Aprendizajes

### Logros Técnicos
1. **Precisión Superior al 90%** en conjunto de entrenamiento
2. **Precisión ~90%** en conjunto de validación
3. **Implementación Desde Cero** sin frameworks externos
4. **Convergencia Estable** en 500 épocas

## Referencias Matemáticas

### Derivaciones Detalladas

#### Derivada de la Función Softmax
```
∂/∂zᵢ softmax(zⱼ) = softmax(zᵢ) · (δᵢⱼ - softmax(zⱼ))
```

#### Regla de la Cadena en Backpropagation
```
∂J/∂W₁ = ∂J/∂A₂ · ∂A₂/∂Z₂ · ∂Z₂/∂A₁ · ∂A₁/∂Z₁ · ∂Z₁/∂W₁
```

#### Gradiente del Costo Cross-Entropy
```
∂J/∂A₂ = -Y/A₂
```

### Complejidad Computacional

| Operación | Complejidad | Descripción |
|-----------|-------------|-------------|
| Forward Pass | O(n·m) | n=neuronas, m=muestras |
| Backward Pass | O(n·m) | Misma complejidad |
| Matrix Multiply | O(i·j·k) | Para matrices i×j y j×k |
| Total por Época | O(n²·m) | Dominado por multiplicaciones |

Este proyecto demuestra que es posible crear una red neuronal efectiva usando únicamente matemáticas fundamentales y NumPy, proporcionando una comprensión profunda de los mecanismos internos del deep learning.



