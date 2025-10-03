Proyecto: Pipeline de MLOps para Predicción de Riesgo Crediticio
1. Planteamiento del Problema
El objetivo de este proyecto es construir un pipeline de Machine Learning de extremo a extremo para predecir si un solicitante de crédito representa un riesgo bueno (clase 1) o malo (clase 2). Al automatizar y optimizar esta predicción, se busca reducir las pérdidas financieras por préstamos incobrables y mejorar la eficiencia del proceso de aprobación, enfocándose en la detección proactiva de clientes de alto riesgo.

2. Dataset
Se utilizó el dataset "Statlog (German Credit Data)" de UCI Machine Learning Repository. Contiene 1000 muestras con 20 características que describen el perfil del solicitante y su historial crediticio.

3. Tecnologías Utilizadas
Lenguaje: Python 3.11

Análisis y Modelado: Pandas, Scikit-learn, Imbalanced-learn

Seguimiento de Experimentos: MLflow

Optimización de Hiperparámetros: Optuna

Despliegue de API: FastAPI, Uvicorn

Containerización y Orquestación: Docker, Prefect

Frontend: HTML, CSS, JavaScript

4. Estructura del Proyecto
El repositorio está organizado siguiendo las mejores prácticas de MLOps:

data/: Contiene los datos crudos del proyecto.

notebooks/: Almacena el Análisis Exploratorio de Datos (EDA).

src/: Contiene todo el código fuente modular de la aplicación.

api.py: Script de la API REST construida con FastAPI para servir predicciones.

index.html: Interfaz de usuario web simple para interactuar con la API.

Dockerfile: Plano para construir la imagen de Docker del proyecto.

prefect_flow.py: Define el flujo de orquestación con Prefect.

requirements.txt: Lista de todas las dependencias de Python.

5. Metodología y Pipeline
Se construyó un pipeline automatizado que ejecuta los siguientes pasos:

Carga de Datos: Un módulo etl.py carga y prepara los datos.

Ingeniería de Características: El módulo feature_engineer.py crea nuevas variables para mejorar el rendimiento.

Optimización y Entrenamiento: Se utiliza Optuna para realizar una búsqueda automática de hiperparámetros. El modelo seleccionado fue BalancedRandomForestClassifier de la librería imbalanced-learn, elegido específicamente para combatir el desbalance de clases del dataset. El objetivo de la optimización fue maximizar la métrica recall para la clase "Mal Riesgo".

Seguimiento de Experimentos: Se integra MLflow para registrar los parámetros y métricas de cada experimento, así como para guardar el artefacto del modelo final.

Validación Robusta: Se implementa Validación Cruzada durante la optimización para obtener una estimación fiable y estable del rendimiento del modelo.

6. Resultados Finales
Tras un proceso de optimización enfocado en la métrica de negocio más importante (detectar malos créditos), el modelo final alcanzó los siguientes resultados en el conjunto de prueba:

Recall (para "Mal Riesgo"): 71.2%

Accuracy (General): 76.5%

Esto representa una mejora sustancial sobre el recall inicial de ~58%, logrando el objetivo de construir un modelo mucho más efectivo para detectar clientes de riesgo. Aunque el accuracy general es moderado, el alto recall lo convierte en una herramienta valiosa para la mitigación de riesgos.

Estudio
Gem personalizado

Ver razonamiento
¡Absolutamente! Es el paso final y más importante para que tu proyecto sea profesional y fácil de entender. Tienes toda la razón, el README actual se ha quedado atrás con todo lo que hemos avanzado.

Lo he reestructurado y actualizado por completo para que refleje la versión final y más potente de tu proyecto, incluyendo las mejoras en el modelo, la API y las instrucciones de uso.

Aquí tienes la versión final recomendada. Puedes reemplazar todo el contenido de tu archivo README.md con este texto.

Proyecto: Pipeline de MLOps para Predicción de Riesgo Crediticio
1. Planteamiento del Problema
El objetivo de este proyecto es construir un pipeline de Machine Learning de extremo a extremo para predecir si un solicitante de crédito representa un riesgo bueno (clase 1) o malo (clase 2). Al automatizar y optimizar esta predicción, se busca reducir las pérdidas financieras por préstamos incobrables y mejorar la eficiencia del proceso de aprobación, enfocándose en la detección proactiva de clientes de alto riesgo.

2. Dataset
Se utilizó el dataset "Statlog (German Credit Data)" de UCI Machine Learning Repository. Contiene 1000 muestras con 20 características que describen el perfil del solicitante y su historial crediticio.

3. Tecnologías Utilizadas
Lenguaje: Python 3.11

Análisis y Modelado: Pandas, Scikit-learn, Imbalanced-learn

Seguimiento de Experimentos: MLflow

Optimización de Hiperparámetros: Optuna

Despliegue de API: FastAPI, Uvicorn

Containerización y Orquestación: Docker, Prefect

Frontend: HTML, CSS, JavaScript

4. Estructura del Proyecto
El repositorio está organizado siguiendo las mejores prácticas de MLOps:

data/: Contiene los datos crudos del proyecto.

notebooks/: Almacena el Análisis Exploratorio de Datos (EDA).

src/: Contiene todo el código fuente modular de la aplicación.

api.py: Script de la API REST construida con FastAPI para servir predicciones.

index.html: Interfaz de usuario web simple para interactuar con la API.

Dockerfile: Plano para construir la imagen de Docker del proyecto.

prefect_flow.py: Define el flujo de orquestación con Prefect.

requirements.txt: Lista de todas las dependencias de Python.

5. Metodología y Pipeline
Se construyó un pipeline automatizado que ejecuta los siguientes pasos:

Carga de Datos: Un módulo etl.py carga y prepara los datos.

Ingeniería de Características: El módulo feature_engineer.py crea nuevas variables para mejorar el rendimiento.

Optimización y Entrenamiento: Se utiliza Optuna para realizar una búsqueda automática de hiperparámetros. El modelo seleccionado fue BalancedRandomForestClassifier de la librería imbalanced-learn, elegido específicamente para combatir el desbalance de clases del dataset. El objetivo de la optimización fue maximizar la métrica recall para la clase "Mal Riesgo".

Seguimiento de Experimentos: Se integra MLflow para registrar los parámetros y métricas de cada experimento, así como para guardar el artefacto del modelo final.

Validación Robusta: Se implementa Validación Cruzada durante la optimización para obtener una estimación fiable y estable del rendimiento del modelo.

6. Resultados Finales
Tras un proceso de optimización enfocado en la métrica de negocio más importante (detectar malos créditos), el modelo final alcanzó los siguientes resultados en el conjunto de prueba:

Recall (para "Mal Riesgo"): 71.2%

Accuracy (General): 76.5%

Esto representa una mejora sustancial sobre el recall inicial de ~58%, logrando el objetivo de construir un modelo mucho más efectivo para detectar clientes de riesgo. Aunque el accuracy general es moderado, el alto recall lo convierte en una herramienta valiosa para la mitigación de riesgos.

7. Instalación y Configuración

# 1. Clona el repositorio
git clone [URL-DE-TU-REPO]
cd [NOMBRE-DEL-REPO]

# 2. Crea y activa un entorno virtual
python -m venv .venv
# En Windows:
.\.venv\Scripts\Activate.ps1
# En macOS/Linux:
# source .venv/bin/activate

# 3. Instala las dependencias
pip install -r requirements.txt

8. Modo de Uso

Para Entrenar el Modelo
Ejecuta el pipeline completo. Esto buscará los mejores hiperparámetros, entrenará el modelo final y lo guardará en la carpeta mlruns/ y models/.

python -m src.app.train.task_train

# Para Visualizar los Experimentos

Inicia la interfaz de usuario de MLflow para explorar los resultados de los entrenamientos.

# Para Ejecutar la API

Inicia el servidor de la API para poder hacer predicciones en tiempo real.

uvicorn api:app --reload
El servidor estará disponible en http://127.0.0.1:8000

# Para Usar la Interfaz Web

Con la API corriendo, abre el archivo index.html en tu navegador para interactuar con el modelo a través de un formulario.
