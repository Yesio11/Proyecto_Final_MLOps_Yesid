import pandas as pd
import mlflow
import optuna
import numpy as np # NUEVO
from sklearn.model_selection import train_test_split, cross_val_score # NUEVO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TrainMlflowOptuna:
    def __init__(self, df: pd.DataFrame, target_column: str, n_trials: int = 25):
        self.df = df
        self.target_column = target_column
        self.n_trials = n_trials

    def _objective(self, trial, X_train, y_train): # NUEVO: Ya no necesitamos X_test, y_test aquí
        """
        Función objetivo que Optuna intentará maximizar usando Validación Cruzada.
        """
        # Espacio de búsqueda de hiperparámetros para RandomForest
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 4, 16)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        # Identificamos columnas
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns
        
        # Creamos el pipeline de preprocesamiento
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Creamos el pipeline con el modelo RandomForestClassifier
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # NUEVO: Usamos validación cruzada con 5 divisiones (folds)
        # Esto entrena y evalúa el modelo 5 veces en diferentes subconjuntos de los datos de entrenamiento
        scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        # Devolvemos el promedio de los scores
        return np.mean(scores)

    def train(self):
        print("Iniciando optimización con Validación Cruzada, Optuna y MLflow...")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        # Todavía hacemos un split inicial para tener un conjunto de prueba final si lo necesitáramos,
        # pero la optimización se hará solo sobre el conjunto de entrenamiento (X_train, y_train).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        study = optuna.create_study(direction='maximize')
        with mlflow.start_run(run_name='Optuna_Parent_Run_CV_RF') as parent_run:
            study.optimize(
                lambda trial: self._objective_with_mlflow(trial, X_train, y_train, parent_run.info.run_id), 
                n_trials=self.n_trials
            )
        print(f"\nOptimización completada. Mejor accuracy (promedio CV): {study.best_value:.4f}")
        print(f"Mejores hiperparámetros: {study.best_params}")
        return study.best_params, study.best_value

    def _objective_with_mlflow(self, trial, X_train, y_train, parent_run_id):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True, experiment_id=mlflow.get_experiment_by_name("Default").experiment_id) as run:
            # NUEVO: Pasamos solo los datos de entrenamiento
            accuracy = self._objective(trial, X_train, y_train)
            mlflow.log_metric("accuracy_cv_mean", accuracy) # Nombramos la métrica para que sea claro que es de CV
            return accuracy