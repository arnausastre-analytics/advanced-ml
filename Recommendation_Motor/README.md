# 🤖 Deep Learning Recommender System – Simulated Ecommerce Dataset

Este proyecto demuestra cómo construir un **motor de recomendación avanzado** utilizando técnicas modernas de Deep Learning (Neural Collaborative Filtering) con `TensorFlow/Keras`, incluso cuando no se dispone de datos reales.

---

## 💡 ¿Qué resuelve este proyecto?

Las recomendaciones inteligentes son clave para:
- Aumentar ventas y retención en ecommerce.
- Personalizar la experiencia del cliente.
- Implementar motores de recomendación similares a los de Amazon, Netflix o Spotify.

Este proyecto simula un entorno realista donde un modelo de deep learning **aprende a predecir qué productos le interesan a cada usuario**.

---

## 🧠 Metodología

### ✅ Dataset Simulado
- 1,000 usuarios  
- 500 productos  
- 10,000 interacciones positivas (compró)  
- 10,000 negativas (no compró) generadas por muestreo

Se genera un dataset balanceado de tipo **usuario-producto → 0 o 1** para entrenar el modelo.

### ✅ Modelo de Recomendación
- **Neural Collaborative Filtering (NCF)** con embeddings de usuarios y productos.
- Capas densas no lineales para capturar relaciones complejas.
- Pérdida binaria + métrica AUC para evaluar rendimiento.

### ✅ Métricas personalizadas
- `Precision@10`  
- `Recall@10`  
- Evaluación por usuario basada en sus compras reales simuladas.

---

## 📊 Análisis avanzado

Además del entrenamiento del modelo, se implementan:

### 🔍 Visualización de embeddings
- Reducción de dimensionalidad con **PCA**.
- Visualización del "mapa latente" de productos en 2D.

### 🎯 Clustering de productos
- Clustering sobre los embeddings usando **K-Means**.
- Identificación de grupos de productos similares en el espacio vectorial.

---

## 📈 Resultados

Ejemplo de recomendación para el usuario `42`:

product_id score
287 0.7950
355 0.7721
... ...


Este sistema sugiere productos personalizados al usuario según su historial y los patrones aprendidos.

Ejemplo de métricas top-N:


---

## ⚙️ Tecnologías usadas

- Python 3  
- `TensorFlow / Keras`  
- `NumPy / Pandas / Scikit-learn`  
- `Matplotlib / PCA / KMeans`

---

## 📁 Estructura del proyecto

├── recommender_simulation.ipynb # Notebook principal
├── requirements.txt # Dependencias (opcional)
└── README.md # Este archivo


---

## 🧠 ¿Por qué este proyecto es valioso?

- Demuestra capacidad para construir modelos reales sin necesidad de datasets externos.
- Integra técnicas de Machine Learning + Deep Learning + Visualización.
- Reutilizable en entornos reales (con solo cambiar el dataset).
- Útil para aplicaciones en:
  - Ecommerce
  - Retail digital
  - Apps de fidelización
  - Sistemas de suscripción

---

## 🚀 Cómo ejecutar

1. Abre el notebook en [Google Colab](https://colab.research.google.com/).
2. Ejecuta todas las celdas paso a paso.
3. Ajusta parámetros, número de usuarios, productos o arquitectura para experimentar.

---

## 📬 Contacto

¿Te interesa integrar un sistema de recomendación personalizado en tu negocio o startup?

**[Tu Nombre]** – Freelance Data Scientist  
📧 tu.email@ejemplo.com  
🔗 [Tu LinkedIn]

---
