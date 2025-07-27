# 🧠 Deep Anomaly Detection in Transactional Data with Autoencoders

Este proyecto implementa un sistema de **detección de anomalías transaccionales** utilizando **Autoencoders con Keras**, ideal para entornos como fintechs, ecommerce o logística.

Detectar comportamientos inusuales —como fraude, errores de sistema o patrones atípicos— puede generar un gran impacto económico. Aquí mostramos cómo hacerlo de forma escalable, no supervisada y visualmente explicable.

---

## 💼 ¿Qué resuelve este proyecto?

> “Monitorizamos tus datos transaccionales y detectamos comportamientos sospechosos que podrían ser fraude o errores operativos, utilizando inteligencia artificial no supervisada.”

---

## ⚙️ Tecnologías utilizadas

- `Python`, `NumPy`, `Pandas`
- `TensorFlow/Keras`
- `Scikit-learn`, `Seaborn`, `Matplotlib`

---

## 🧠 Metodología

### ✅ 1. Simulación de Datos Realistas
- Transacciones normales: montos regulares, descuentos típicos, compras comunes.
- Anomalías simuladas: compras inusualmente altas, grandes descuentos, comportamiento extraño.
- Etiquetado binario: 0 = normal, 1 = anómala (solo para evaluación).

### ✅ 2. Preprocesamiento
- Escalado estándar de variables.
- División en datos de entrenamiento y prueba.

### ✅ 3. Autoencoder Profundo
- Arquitectura simétrica: codificador + decodificador.
- Entrenamiento para reconstruir transacciones normales.
- **Anomalías = transacciones con alto error de reconstrucción.**

### ✅ 4. Umbral de detección
- Umbral automático basado en el percentil 95 del error (MSE).
- Detección binaria de anomalías a partir del umbral.

---

## 📈 Resultados

### 🧪 Métricas clave

Matriz de confusión:
[[4866 134]
[ 26 124]]

📈 Reporte de clasificación:
Accuracy: 96.9%
Precision (1): 48.1%
Recall (1): 82.7%
F1-score (1): 60.8%


### 📊 Visualizaciones incluidas

- Distribución de error de reconstrucción
- Boxplot por tipo de transacción
- Top 10 transacciones más sospechosas
- Curva ROC (AUC)
- Curva Precision-Recall (PR AUC)
- Scatterplot de anomalías sobre variables clave

---

## 📍 Estructura del proyecto

├── anomaly_autoencoder.ipynb # Notebook principal con todo el pipeline
├── README.md # Este archivo


---

## 🧠 Valor del proyecto

Este sistema puede aplicarse directamente a:

- 🔐 **Detección de fraude** financiero, bancario, criptomonedas.
- 📦 **Errores en pedidos logísticos** o distribución.
- 🛒 **Análisis de calidad de datos** en ecommerce y ventas.
- 🧾 Auditoría interna y control de integridad.

El sistema es **no supervisado**, escalable y adaptable a cualquier conjunto transaccional con múltiples variables numéricas o categóricas.

---

## 🚀 Cómo usar

1. Abre el notebook en Google Colab.
2. Ejecuta todas las celdas.
3. Reemplaza el dataset simulado por tus propios datos transaccionales.

---

## 📬 Contacto

¿Quieres integrar un sistema de detección de anomalías en tiempo real en tu empresa?

**[Tu Nombre]** — Freelance Data Scientist  
📧 tu.email@ejemplo.com  
🔗 [Tu LinkedIn]

---
