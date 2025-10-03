import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

# ... (las otras importaciones) ...

# --- NUEVO: Especificar la RUTA ABSOLUTA a los experimentos de MLflow ---
# Asegúrate de poner una 'r' antes de las comillas si estás en Windows.
# REEMPLAZA el texto de adentro con la ruta que copiaste.
MLRUNS_PATH = r'C:\Users\USER\Documents\repositorios\Proyecto_Final_MLOps_Yesid\mlruns'
mlflow.set_tracking_uri(f"file:///{MLRUNS_PATH}")
# --------------------------------------------------------------------

# ... (el resto del código permanece igual) ...

# 1. Definir el modelo de datos de entrada con Pydantic
# Estas son las características que nuestra API esperará recibir.
# Deben coincidir con las columnas que usó el modelo para entrenar.
class CreditApplicant(BaseModel):
    estado_cuenta_actual: str
    duracion_credito_meses: int
    historial_credito: str
    proposito_credito: str
    monto_credito: int
    ahorros_o_bonos: str
    tiempo_empleo: str
    tasa_de_cuotas: int
    estado_civil_y_sexo: str
    otros_deudores_o_garantes: str
    tiempo_residencia_actual: int
    propiedad: str
    edad: int
    otros_planes_cuotas: str
    vivienda: str
    numero_creditos_existentes: int
    trabajo: str
    numero_personas_dependientes: int
    telefono: str
    trabajador_extranjero: str
    monto_por_mes: float
    proporcion_credito_edad: float

# 2. Iniciar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Riesgo Crediticio", version="1.0")

# 3. Cargar el mejor modelo de MLflow al iniciar la API
# Esta variable global contendrá nuestro modelo una vez cargado.
model = None

@app.on_event("startup")
def load_model():
    """
    Función que se ejecuta al iniciar la API.
    Busca el mejor run en MLflow y carga el modelo convirtiendo la URI a una ruta local.
    """
    global model
    # NUEVO: Importaciones adicionales para manejar las rutas
    from urllib.parse import urlparse
    import os

    print("Cargando el mejor modelo desde MLflow...")
    
    # Aseguramos que MLflow sepa dónde buscar
    mlflow.set_tracking_uri("mlruns")

    best_run = mlflow.search_runs(
        experiment_names=["Default"],
        order_by=["metrics.recall_cv_mean DESC"],
        max_results=1
    )
    
    if best_run.empty:
        raise RuntimeError("No se encontraron runs de MLflow.")
        
    # Obtenemos la URI del artefacto (ej: 'file:///C:/path/to/mlruns/...')
    artifact_uri = best_run.iloc[0].artifact_uri
    run_id = best_run.iloc[0].run_id

    # --- NUEVO: Convertimos la URI a una ruta de sistema de archivos ---
    # Parsea la URI para obtener la ruta y elimina la barra inicial
    model_path = urlparse(artifact_uri).path.lstrip('/')
    # Le añadimos la subcarpeta 'model'
    model_path = os.path.join(model_path, "model")
    
    print(f"Mejor run_id encontrado: {run_id}")
    print(f"Ruta de modelo convertida a: {model_path}")

    # Cargamos el modelo usando la ruta de sistema de archivos directa
    model = mlflow.pyfunc.load_model(model_path)
    print("Modelo cargado exitosamente.")

# 4. Crear el endpoint de predicción
@app.post("/predict")
def predict(applicant_data: CreditApplicant):
    """
    Recibe los datos de un solicitante, realiza la predicción y la devuelve.
    """
    if model is None:
        return {"error": "El modelo no está cargado."}

    # Convertir los datos de Pydantic a un DataFrame de Pandas
    df = pd.DataFrame([applicant_data.dict()])
    
    # Realizar la predicción
    # MLflow se encarga de todo el preprocesamiento definido en el pipeline
    prediction = model.predict(df)
    
    # Convertir el resultado a un formato legible
    # Recordatorio: Clase 1 = Buen Riesgo, Clase 2 = Mal Riesgo
    riesgo = "Buen Riesgo" if prediction[0] == 1 else "Mal Riesgo"
    
    return {"prediccion_riesgo": riesgo}