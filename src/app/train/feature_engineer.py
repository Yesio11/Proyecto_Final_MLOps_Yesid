import pandas as pd

class FeatureEngineer:
    """
    Clase para realizar la ingeniería de características.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el FeatureEngineer con un DataFrame.

        Args:
            df (pd.DataFrame): El DataFrame de entrada.
        """
        self.df = df

    def create_features(self):
        """
        Aplica transformaciones y crea nuevas características.
        """
        print("Iniciando la ingeniería de características...")
        
        # Característica 1: Costo mensual del crédito
        # Se añade un valor pequeño (epsilon) para evitar división por cero
        self.df['monto_por_mes'] = self.df['monto_credito'] / (self.df['duracion_credito_meses'] + 1e-6)
        
        # Característica 2: Proporción del crédito respecto a la edad
        self.df['proporcion_credito_edad'] = self.df['monto_credito'] / (self.df['edad'] + 1e-6)
        
        print("Nuevas características creadas: 'monto_por_mes', 'proporcion_credito_edad'.")
        print("Ingeniería de características completada.")
        return self.df