import pandas as pd
import mlflow
import optuna
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, make_scorer

class TrainMlflowOptuna:
    def __init__(self, df: pd.DataFrame, target_column: str, n_trials: int = 25):
        self.df = df
        self.target_column = target_column
        self.n_trials = n_trials

    def _objective(self, trial, X_train, y_train):
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
        
        # Creamos el pipeline con el modelo BalancedRandomForestClassifier
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', BalancedRandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Usamos validación cruzada con 5 folds y múltiples métricas
        scoring = {
            'accuracy': 'accuracy',
            'recall_bad_risk': make_scorer(recall_score, pos_label=2)
        }
        
        scores = cross_validate(model_pipeline, X_train, y_train, cv=5, scoring=scoring)

        # Log de métricas a MLflow
        trial.set_user_attr("recall_cv_mean", np.mean(scores['test_recall_bad_risk']))

        # Optuna maximizará el accuracy
        return np.mean(scores['test_accuracy'])

    def train(self):
        print("Iniciando optimización con Validación Cruzada, Optuna y MLflow...")
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        study = optuna.create_study(direction='maximize')
        with mlflow.start_run(run_name='Optuna_Parent_Run_CV_BRF') as parent_run: # Changed run name
            study.optimize(
                lambda trial: self._objective_with_mlflow(trial, X_train, y_train, parent_run.info.run_id), 
                n_trials=self.n_trials
            )

        print(f"\nOptimización completada.")
        print(f"Mejor accuracy (promedio CV): {study.best_value:.4f}")

        # Obtenemos el recall del mejor trial
        best_recall = study.best_trial.user_attrs.get("recall_cv_mean", "N/A")
        if isinstance(best_recall, float):
            print(f"Mejor recall para 'Bad Risk' (promedio CV): {best_recall:.4f}")
        else:
            print(f"Recall no disponible para el mejor trial.")

        print(f"Mejores hiperparámetros: {study.best_params}")

        # Entrenar y evaluar el modelo final con los mejores hiperparámetros
        print("\nEntrenando el modelo final con los mejores hiperparámetros en el conjunto de entrenamiento completo...")
        best_params = study.best_params
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        final_model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', BalancedRandomForestClassifier(
                **best_params,
                random_state=42,
                n_jobs=-1
            ))
        ])

        final_model_pipeline.fit(X_train, y_train)
        y_pred = final_model_pipeline.predict(X_test)

        final_accuracy = accuracy_score(y_test, y_pred)
        final_recall = recall_score(y_test, y_pred, pos_label=2)

        print("\n--- Métricas del Modelo Final en el Conjunto de Prueba ---")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Recall para 'Bad Risk' (Clase 2): {final_recall:.4f}")
        print("----------------------------------------------------------")

        return study.best_params, study.best_value

    def _objective_with_mlflow(self, trial, X_train, y_train, parent_run_id):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True, experiment_id=mlflow.get_experiment_by_name("Default").experiment_id) as run:
            accuracy = self._objective(trial, X_train, y_train)
            recall = trial.user_attrs.get("recall_cv_mean", 0) # Recuperamos el recall

            mlflow.log_metric("accuracy_cv_mean", accuracy)
            mlflow.log_metric("recall_cv_mean", recall)
            mlflow.log_params(trial.params)

            return accuracy