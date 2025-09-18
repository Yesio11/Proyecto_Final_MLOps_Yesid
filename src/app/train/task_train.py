# Importamos la clase avanzada final
from src.app.train.etl import DataLoader
from src.app.train.feature_engineer import FeatureEngineer
from src.app.train.train_mlflow_optuna import TrainMlflowOptuna

def main():
    """
    Función principal para ejecutar el pipeline completo de ML con MLOps,
    incluyendo optimización con Validación Cruzada.
    """
    print("Iniciando el pipeline de Machine Learning con Optuna y MLflow...")

    # --- ETAPA DE CARGA DE DATOS (ETL) ---
    DATA_PATH = 'data/raw/statlog+german+credit+data'
    data_loader = DataLoader(data_path=DATA_PATH)
    df = data_loader.load_and_prepare_data()

    if df is not None:
        # --- ETAPA DE INGENIERÍA DE CARACTERÍSTICAS ---
        feature_engineer = FeatureEngineer(df)
        df_engineered = feature_engineer.create_features()
        
        # --- ETAPA DE ENTRENAMIENTO AVANZADO CON VALIDACIÓN CRUZADA ---
        print("\nPasando a la etapa de Entrenamiento con Optuna y MLflow...")
        TARGET_COLUMN = 'clase'
        
        # Creamos una instancia de la clase final
        # Le decimos que realice 25 experimentos (trials)
        advanced_trainer = TrainMlflowOptuna(
            df=df_engineered, 
            target_column=TARGET_COLUMN,
            n_trials=25 
        )
        
        best_params, best_value = advanced_trainer.train()
        
        # --- FIN DEL PIPELINE ---
        print("\n¡Pipeline de MLOps completado exitosamente!")
        print(f"Mejor accuracy (promedio CV) encontrado: {best_value:.4f}")
        print(f"Mejores hiperparámetros: {best_params}")
    else:
        print("\nEl pipeline falló en la etapa de carga de datos.")

if __name__ == '__main__':
    main()