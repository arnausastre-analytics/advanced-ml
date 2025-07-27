# 💸 Dynamic Pricing with Reinforcement Learning (Multi-Armed Bandits)

Este proyecto implementa un sistema de **pricing dinámico basado en aprendizaje por refuerzo (Reinforcement Learning)**. Se simula un entorno de ventas online donde un agente aprende a ajustar los precios para **maximizar ingresos** a través de prueba y error.

---

## 🎯 Objetivo

> “Optimizamos precios dinámicos en tiempo real para maximizar beneficios en marketplaces, ecommerce y plataformas SaaS.”

---

## 🧠 ¿Qué resuelve este sistema?

- Aprende de forma autónoma qué precios generan más ingresos.
- Balancea exploración (probar precios nuevos) y explotación (usar los que funcionan).
- Se adapta a patrones de conversión no lineales y cambiantes.
- Puede integrarse en sistemas reales con mínima supervisión humana.

---

## ⚙️ Tecnologías usadas

- `Python`, `NumPy`, `Pandas`, `Matplotlib`
- Algoritmos de Reinforcement Learning (Bandits)
  - Epsilon-Greedy (básico)
  - Thompson Sampling (avanzado, bayesiano)
- Simulación de entorno comercial con tasas de conversión realistas

---

## 🏗️ Arquitectura del proyecto

Simulación de entorno de ventas:

5 precios posibles

Cada precio tiene su propia tasa de conversión

Agente RL (Bandit) aprende por iteración:

Selecciona precios y observa ingresos

Mejora decisiones con el tiempo

Comparación entre algoritmos:

Ingreso acumulado

Porcentaje de decisiones óptimas


---

## 🤖 Algoritmos implementados

### ✅ Epsilon-Greedy (Multi-Armed Bandit)

- Explora con probabilidad ε
- Explota el precio con mejor media de reward
- Fácil de entender, ideal para MVPs o decisiones rápidas

### ✅ Thompson Sampling (Bayesian Bandit)

- Estima distribuciones beta para cada precio (conversión)
- Muestrea de forma probabilística el mejor precio en cada ronda
- Converge más rápido y genera mayor ROI
- Muy usado en plataformas como Booking, Amazon, YouTube

---

## 📈 Resultados

💰 Revenue real esperado por precio:
Precio: $10 → Conversión: 35.0% → Revenue esperado: 3.50
...
Precio: $50 → Conversión: 5.0% → Revenue esperado: 2.50

🏆 Mejor precio real: $30


### 🔁 Ingreso acumulado tras 1000 rondas

| Algoritmo         | Ingreso total | % de veces que eligió el precio óptimo |
|-------------------|---------------|-----------------------------------------|
| Epsilon-Greedy    | $X.XX         | XX.X%                                   |
| Thompson Sampling | $Y.YY         | YY.Y%                                   |

📊 Gráficos:
- Reward acumulado por precio
- Evolución de ingresos por ronda
- Comparación entre algoritmos

---

## 📁 Estructura del proyecto

├── dynamic_pricing_bandits.ipynb # Notebook completo
├── README.md # Este archivo


---

## 💡 Aplicaciones reales

- Marketplaces: ajusta precios por tipo de cliente, demanda o stock.
- Ecommerce: maximiza beneficio sin comprometer conversión.
- SaaS: pruebas de precios por segmento o canal.
- Publicidad online: optimiza oferta vs click-through rate.

---

## 🚀 Cómo usar

1. Abre el notebook en Google Colab o Jupyter.
2. Ejecuta todas las celdas.
3. Modifica el entorno simulado si quieres probar diferentes curvas de conversión.
4. (Opcional) Sustituye la simulación por datos reales.

---

## 📬 Contacto

¿Quieres implementar un sistema de pricing dinámico real adaptado a tu negocio?

**[Tu Nombre]** – Freelance Data Scientist  
📧 tu.email@ejemplo.com  
🔗 [Tu LinkedIn]

---
