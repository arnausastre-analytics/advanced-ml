# Forecasting Multivariado con Deep Learning (LSTM)

Este proyecto implementa un modelo avanzado de predicción de demanda multivariada usando redes neuronales LSTM. El sistema predice la **demanda futura de productos (multistep)** considerando múltiples variables: precio, promociones, festivos, estacionalidad, etc.

## ¿Qué soluciona?

- Predicción de demanda a 7 días vista (multistep)
- Adaptación a múltiples variables predictoras (precio, promos…)
- Reducción de quiebres de stock y sobreinventario
- Preparación ante picos de demanda (festivos, promociones)
- Visualización y análisis detallado de los errores de predicción

## Tecnologías utilizadas

- `Python`, `NumPy`, `Pandas`
- `TensorFlow / Keras` – LSTM multistep
- `Scikit-learn` – preprocesado y métricas
- `Matplotlib`, `Seaborn` – visualización técnica y explicativa

## Simulación de datos (entorno realista)

Se genera un dataset sintético con estructura realista:

- `price`: valor diario del producto
- `promo`: indicador de promoción activa
- `holiday`: indicador de festivos
- `demand`: serie objetivo (ventas)

Se introducen **componentes de estacionalidad y ruido** para reflejar la realidad operativa de retail o consumo.

## Arquitectura del modelo

- Normalización por separado para `features` y `target`
- Conversión a formato `secuencia → predicción multistep`
- Red neuronal LSTM con capa densa de salida a 7 pasos (t+1...t+7)
- Evaluación por paso de predicción
- Visualización por cliente individual

## Ejemplo de resultados

Datos preparados: X = (1346, 30, 4), y = (1346, 7)
MSE: 28.42, MAE: 4.13

Error por paso:

t+1 → MAE = 3.12

t+4 → MAE = 4.91

t+7 → MAE = 5.88

## Visualizaciones incluidas

- Curva de error por horizonte de predicción (t+1 → t+7)
- Predicciones multistep por cliente
- Gráficos de comparativa real vs. predicho

## Estructura del proyecto

├── forecasting_lstm.ipynb # Notebook principal

├── README.md # Este archivo

## Aplicaciones reales

- Retail: predicción de ventas por SKU o categoría
- Energía: consumo eléctrico por hora/día/cliente
- Logística: predicción de demanda de transporte o stock
- Producción: planificación de turnos según demanda estimada

## Cómo usar

1. Ejecuta el notebook en Google Colab.
2. Visualiza las predicciones multistep.
3. Sustituye el dataset simulado por tus datos reales.
4. Ajusta variables predictoras según tu caso de negocio.
