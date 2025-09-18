# Proyecto: Predicción de Riesgo Crediticio con MLOps

## 1. Planteamiento del Problema
El objetivo de este proyecto es desarrollar un modelo de Machine Learning para predecir si un solicitante de crédito representa un riesgo bueno o malo. Al automatizar y optimizar esta predicción, se busca reducir las pérdidas financieras por préstamos incobrables y mejorar la eficiencia del proceso de aprobación.

## 2. Dataset
Se utilizó el dataset "Statlog (German Credit Data)" de UCI Machine Learning Repository. Contiene 1000 muestras con 20 características categóricas y numéricas que describen el perfil del solicitante y su historial crediticio.

## 3. Estructura del Proyecto
El repositorio está organizado siguiendo las mejores prácticas de MLOps:
- **/data**: Contiene los datos brutos del proyecto.
- **/notebooks**: Almacena los notebooks de análisis exploratorio inicial (EDA).
- **/src**: Contiene todo el código fuente modular de la aplicación.
  - **/src/app/train**: Módulos para ETL, ingeniería de características y entrenamiento de modelos.

## 4. Metodología y Pipeline
Se construyó un pipeline automatizado que ejecuta los siguientes pasos:
1.  **Carga de Datos**: Un módulo `etl.py` carga y prepara los datos.
2.  **Ingeniería de Características**: El módulo `feature_engineer.py` crea nuevas variables (`monto_por_mes`, etc.) para mejorar el rendimiento.
3.  **Optimización y Entrenamiento**: Se utiliza `Optuna` para realizar una búsqueda automática de hiperparámetros sobre un modelo `RandomForestClassifier`.
4.  **Seguimiento de Experimentos**: Se integra `MLflow` para registrar cada experimento, sus parámetros y métricas.
5.  **Validación Robusta**: Se implementa **Validación Cruzada (Cross-Validation)** para obtener una estimación fiable del rendimiento del modelo.

## 5. Resultados
El pipeline de optimización se ejecutó con un modelo RandomForestClassifier. Mediante una búsqueda de hiperparámetros con Optuna y una evaluación robusta con Validación Cruzada de 5 folds, el modelo final alcanzó un accuracy promedio de 77%. Los mejores hiperparámetros encontrados fueron:

- **n_estimators**: [204]
- **max_depth**: [16]
- **min_samples_split**: [3]

Aunque el accuracy es un buen punto de partida, un análisis más profundo es crucial para un problema de riesgo crediticio. Un análisis de la matriz de confusión y las métricas de clasificación reveló que el recall para la clase "Mal Riesgo" es de aproximadamente 58%. Esto significa que el modelo solo identifica correctamente a 6 de cada 10 solicitantes que representan un mal riesgo, dejando pasar al 42% restante.

Conclusión: Debido al alto riesgo financiero que representan los falsos negativos (malos créditos aprobados), se concluye que el modelo en su estado actual no es adecuado para un despliegue totalmente automático. Sin embargo, representa una base sólida y podría ser utilizado como una herramienta de apoyo para analistas humanos.