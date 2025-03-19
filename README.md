# Company Bankruptcy Prediction

Este proyecto tiene como objetivo **predecir la quiebra de empresas** utilizando técnicas de Machine Learning. Se ha trabajado con un conjunto de datos extraído del *Taiwan Economic Journal* para el período 1999–2009, donde la quiebra de una empresa se define según las regulaciones comerciales de la Bolsa de Valores de Taiwán.

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Dataset](#dataset)
- [Requisitos](#requisitos)
- [Ejecución del Proyecto](#ejecución-del-proyecto)
- [Estructura del Código](#estructura-del-código)
- [Evaluación y Métricas](#evaluación-y-métricas)
- [Guardado de Modelos](#guardado-de-modelos)
- [Futuras Mejoras](#futuras-mejoras)
- [Referencias](#referencias)

---

## Descripción del Proyecto

El propósito de este proyecto es desarrollar un modelo predictivo para identificar empresas en riesgo de quiebra. Se realizaron las siguientes tareas principales:

- **Carga y preprocesamiento de datos:** Lectura del dataset, manejo del desbalanceo de clases mediante SMOTE.
- **Entrenamiento de modelos:** Se entrenaron dos modelos:
  - **RandomForestClassifier** con 400 estimadores.
  - **XGBClassifier** utilizando XGBoost.
- **Evaluación:** Se calcularon métricas de rendimiento como exactitud, F1-score, recall, y se generaron curvas ROC y matrices de confusión para analizar el comportamiento del modelo.
- **Serialización de Modelos:** Guardado de los modelos entrenados mediante `joblib` para su reutilización futura.

---

## Dataset

El dataset se compone de datos financieros recopilados del *Taiwan Economic Journal* (1999–2009). Se define la quiebra de una empresa con base en las regulaciones de la Bolsa de Valores de Taiwán.  

**Atributos principales:**

- **Y - Bankrupt?**  
  Etiqueta de clase (1 si la empresa quiebra, 0 si no).
  
- **X1 a X95:**  
  Variables de entrada que incluyen métricas financieras como:
  - **X1:** ROA(C) (Rentabilidad sobre activos totales, modalidad C)
  - **X2:** ROA(A) (Rentabilidad sobre activos totales, modalidad A)
  - **X3:** ROA(B) (Rentabilidad sobre activos totales, modalidad B)
  - **X4:** Operating Gross Margin (Margen bruto operativo)
  - **X5:** Realized Sales Gross Margin (Margen bruto realizado)
  - ...
  - **X92:** Degree of Financial Leverage (DFL)
  - **X93:** Interest Coverage Ratio (Cobertura de intereses)
  - **X94:** Net Income Flag (Indicador de ingresos netos negativos en los últimos 2 años)
  - **X95:** Equity to Liability (Relación entre patrimonio y pasivo)

Se pueden consultar más detalles y la fuente completa en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction) o en la plataforma Kaggle, de donde fue extraído el dataset: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/data)

---

## Requisitos

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes librerías de Python:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

---

## Ejecución del Proyecto

1. **Carga del dataset:**  
   El archivo `data.csv` contiene los datos originales.  
2. **Preprocesamiento:**  
   Se divide el dataset en entrenamiento y prueba, y se aplica SMOTE para balancear la clase minoritaria.
3. **Entrenamiento:**  
   Se entrena un modelo `RandomForestClassifier` y un modelo `XGBClassifier` utilizando los datos balanceados.
4. **Evaluación:**  
   Se calculan métricas como Accuracy, F1-score, Recall y se generan la curva ROC y la matriz de confusión.
5. **Guardado:**  
   Los modelos entrenados se guardan en archivos usando `joblib`.

---

## Evaluación y Métricas

- **Accuracy:** Exactitud global del modelo.
- **F1-score:** Equilibrio entre precisión y recall, especialmente útil en datasets desbalanceados.
- **Recall (Exhaustividad):** Capacidad del modelo para identificar correctamente los casos positivos.
- **Curva ROC y AUC:** Para evaluar la capacidad del modelo de distinguir entre las clases.
- **Matriz de Confusión:** Se visualiza la distribución de verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos.

---

## Guardado de Modelos

```python
import joblib

joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(xgb_classifier, "xgb_classifier.pkl")
```

Para verificar que se guardaron correctamente:

```python
rf_loaded = joblib.load("random_forest_model.pkl")
print(np.array_equal(rf.predict(x_test), rf_loaded.predict(x_test)))  # Debe devolver True
```

---

## Futuras Mejoras

- **Ajustar el umbral de decisión.**
- **Explorar otros algoritmos o ensembles** como Gradient Boosting, LightGBM o CatBoost.
- **Optimizar hiperparámetros** con GridSearch o RandomSearch.
- **Feature Engineering adicional** para mejorar la separabilidad de clases.

---

## Referencias

- [Taiwanese Bankruptcy Prediction (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction)
- Liang, D., Lu, C.-C., Tsai, C.-F., and Shih, G.-A. (2016). *Financial Ratios and Corporate Governance Indicators in Bankruptcy Prediction: A Comprehensive Study*. European Journal of Operational Research, 252(2), 561-572.
- **Bibliotecas utilizadas:** pandas, scikit-learn, xgboost, imbalanced-learn, joblib, seaborn, matplotlib.
