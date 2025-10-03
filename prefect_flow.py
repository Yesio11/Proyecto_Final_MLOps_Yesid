from prefect import task, flow
# NUEVO: Importamos el bloque de Docker
from prefect_docker import DockerContainer

from src.app.train.etl import DataLoader
from src.app.train.feature_engineer import FeatureEngineer
from src.app.train.train_mlflow_optuna import TrainMlflowOptuna
import pandas as pd

@task(name="Cargar Datos")
def load_data_task() -> pd.DataFrame:
    DATA_PATH = 'data/raw/statlog+german+credit+data'
    data_loader = DataLoader(data_path=DATA_PATH)
    df = data_loader.load_and_prepare_data()
    return df

@task(name="Ingeniería de Características")
def feature_engineer_task(df: pd.DataFrame) -> pd.DataFrame:
    feature_engineer = FeatureEngineer(df)
    df_engineered = feature_engineer.create_features()
    return df_engineered

@task(name="Entrenar Modelo con MLOps")
def train_model_task(df: pd.DataFrame):
    TARGET_COLUMN = 'clase'
    advanced_trainer = TrainMlflowOptuna(
        df=df, 
        target_column=TARGET_COLUMN,
        n_trials=25
    )
    best_params, best_value = advanced_trainer.train()
    print(f"Mejor accuracy (promedio CV) encontrado: {best_value:.4f}")
    print(f"Mejores hiperparámetros: {best_params}")

# NUEVO: Definimos el bloque de Docker que Prefect usará
docker_block = DockerContainer(
    image="mi-proyecto-mlops:latest", # El nombre de la imagen que construimos
    auto_remove=True, # Limpia el contenedor después de que termina
)

@flow(name="Pipeline de Entrenamiento Dockerizado", flow_runner=docker_block)
def main_flow():
    """
    Flujo principal que ahora se ejecuta dentro de un contenedor Docker.
    """
    df = load_data_task()
    df_engineered = feature_engineer_task(df)
    train_model_task(df_engineered)

# NOTA: Ya no necesitamos el bloque if __name__ == "__main__"
# porque ahora Prefect gestionará la ejecución.