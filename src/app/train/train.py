import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Train:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target_column = target_column
        self.model = None

    def train(self):
        print("Iniciando el proceso de entrenamiento base...")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(random_state=42))
        ])

        self.model.fit(X_train, y_train)
        print("Modelo entrenado exitosamente.")

        # --- SECCIÓN DE EVALUACIÓN DETALLADA ---
        print("\n--- Evaluación Detallada del Modelo ---")
        y_pred = self.model.predict(X_test)
        
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        print("\nMatriz de Confusión:")
        # Recordatorio: La clase 1 es 'Buen Riesgo', la 2 es 'Mal Riesgo'
        print(confusion_matrix(y_test, y_pred))
        
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred, target_names=['Buen Riesgo (1)', 'Mal Riesgo (2)']))
        
        return self.model