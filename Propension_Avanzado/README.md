# 🛍️ Propensity to Buy Model with XGBoost + SHAP

Este proyecto implementa un sistema avanzado para **predecir la probabilidad de compra** de clientes usando técnicas de machine learning y análisis interpretativo con SHAP.

---

## 🎯 Objetivo del proyecto

> “Identificamos qué clientes tienen más probabilidad de convertir, y explicamos por qué, para optimizar campañas de marketing, remarketing o fidelización.”

---

## 🧠 ¿Qué resuelve este sistema?

- Priorización de clientes con mayor **propensión a comprar**
- Optimización del **umbral de conversión** para maximizar resultados
- Interpretabilidad a nivel **individual y global**
- Simulación del **retorno de inversión (ROI)** de campañas basadas en el modelo
- Modelos específicos por segmento de cliente

---

## ⚙️ Tecnologías utilizadas

- `Python`, `NumPy`, `Pandas`
- `XGBoost` – modelo de clasificación
- `SHAP` – interpretabilidad avanzada
- `Scikit-learn` – métricas y evaluación
- `Matplotlib` / `Plotly` – visualizaciones

---

## 📊 Metodología

### ✅ Simulación realista de datos de comportamiento
Variables simuladas:
- Visitas web, clics en emails, productos vistos
- Descuento aplicado, frecuencia del cliente
- Segmento de marketing

### ✅ Entrenamiento de modelo XGBoost
- Clasificación binaria: `compró` vs. `no compró`
- Separación entrenamiento/test
- Evaluación con métricas de negocio (F1, AUC)

### ✅ Interpretabilidad con SHAP
- Importancia global de variables (summary plot)
- Contribuciones individuales (waterfall plot)
- Exportación de un `reporte SHAP` por cliente

### ✅ Optimización del umbral
- Selección del mejor umbral para maximizar **F1-score**
- Evaluación detallada: `precision`, `recall`, `accuracy`, `AUC`

### ✅ Simulación de campaña y ROI
- Aplicación del modelo al 10% más propenso
- Cálculo de conversión real + ROI estimado de una acción de marketing

### ✅ Modelos personalizados por segmento
- Entrenamiento y evaluación separados por tipo de cliente
- Permite estrategias diferenciadas de targeting

---

## 📈 Ejemplo de resultados

📌 Umbral óptimo (F1): 0.43 → F1 = 0.7912

💰 ROI estimado en campaña sobre top 10% más propensos: 5.20x
🎯 Tasa de conversión real del top 10%: 42.00%

📤 Reporte SHAP exportado: reporte_shap_clientes.csv

📊 Entrenamiento por segmento:

Segmento 0: AUC = 0.91, Clientes = 2500

Segmento 1: AUC = 0.93, Clientes = 1500

Segmento 2: AUC = 0.95, Clientes = 1000


---

## 📁 Estructura del proyecto

├── propensity_model.ipynb # Notebook principal completo
├── reporte_shap_clientes.csv # Reporte exportado (predicciones + SHAP)
├── README.md # Este archivo


---

## 🧩 Aplicaciones reales

- Campañas de marketing predictivo
- Segmentación y scoring de clientes
- Automatización de retargeting
- CRM inteligente y acciones en tiempo real
- Fidelización y churn prediction

---

## 🚀 Cómo usar

1. Abre el notebook en Google Colab.
2. Ejecuta todas las celdas.
3. Reemplaza el dataset simulado por tus propios datos de clientes.
4. Personaliza variables, ROI, umbral o segmentación según tus necesidades.

---

## 📬 Contacto

¿Te interesa implementar un sistema de scoring y targeting personalizado para tus clientes?

**[Tu Nombre]** – Freelance Data Scientist  
📧 tu.email@ejemplo.com  
🔗 [Tu LinkedIn]

---
