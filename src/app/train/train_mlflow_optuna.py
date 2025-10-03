import pandas as pd
import mlflow
import optuna
import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

class TrainMlflowOptuna:
    def __init__(self, df: pd.DataFrame, target_column: str, n_trials: int = 25):
        self.df = df
        self.target_column = target_column
        self.n_trials = n_trials

    def _objective(self, trial, X_train, y_train):
        # Esta función solo se usa para la búsqueda de parámetros
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 4, 16),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'random_state': 42,
            'n_jobs': 1 # Usamos 1 para evitar problemas en Windows
        }
        
        model = BalancedRandomForestClassifier(**params)
        
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
            ])
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # Optimizamos usando la métrica recall
        score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='recall_macro').mean()
        return score

    def train(self):
        print("--- Fase 1: Búsqueda de Hiperparámetros con Optuna ---")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)

        best_params = study.best_params
        best_score = study.best_value
        print(f"\nBúsqueda completada. Mejor recall (promedio CV): {best_score:.4f}")
        print(f"Mejores hiperparámetros encontrados: {best_params}")

        # --- Fase 2: Entrenamiento y Guardado del Modelo Final ---
        print("\n--- Fase 2: Entrenando el modelo final con los mejores parámetros... ---")
        with mlflow.start_run(run_name="Final_Model") as run:
            # Log de los mejores parámetros y la métrica
            mlflow.log_params(best_params)
            mlflow.log_metric("best_recall_cv", best_score)

            # Creamos el pipeline final con los mejores parámetros
            final_model = BalancedRandomForestClassifier(**best_params, random_state=42, n_jobs=1)
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object']).columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
                ])
            final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_model)])

            # Entrenamos el modelo final con TODOS los datos
            final_pipeline.fit(X, y)

            # Guardamos el modelo final como un artefacto en MLflow
            mlflow.sklearn.log_model(
                sk_model=final_pipeline,
                artifact_path="model",
                input_example=X.head(5)
            )
            
            
            print("Modelo final entrenado y guardado en MLflow exitosamente.")

            print("Guardando el modelo en la carpeta local 'models'...")
            os.makedirs('models', exist_ok=True) # Crea la carpeta si no existe
            joblib.dump(final_pipeline, 'models/credit_risk_model.joblib')
            print("Modelo guardado exitosamente como 'credit_risk_model.joblib'.")
            # ---------------------------------------------------------
        
        return best_params, study.best_value