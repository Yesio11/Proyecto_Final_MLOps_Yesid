import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 

# Definimos la estructura de los datos de entrada
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
    monto_por_mes: float = 0.0
    proporcion_credito_edad: float = 0.0

# Iniciar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Riesgo Crediticio", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todas las cabeceras
)
# Variable global para almacenar el modelo cargado
model = None

@app.on_event("startup")
def load_model():
    """
    Al iniciar la API, carga el modelo directamente desde el archivo .joblib.
    """
    global model
    print("Cargando modelo desde 'models/credit_risk_model.joblib'...")
    model = joblib.load('models/credit_risk_model.joblib')
    print("Modelo cargado exitosamente.")

@app.post("/predict")
def predict(applicant_data: CreditApplicant):
    if model is None:
        return {"error": "El modelo no está cargado."}

    df = pd.DataFrame([applicant_data.dict()])
    prediction = model.predict(df)
    riesgo = "Buen Riesgo" if prediction[0] == 1 else "Mal Riesgo"
    
    return {"prediccion_riesgo": riesgo}