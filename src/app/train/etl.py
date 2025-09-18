import pandas as pd
import os

class DataLoader:
    """
    Clase para cargar y preparar los datos del German Credit Dataset.
    """
    def __init__(self, data_path):
        """
        Inicializa el DataLoader con la ruta a la carpeta de datos.

        Args:
            data_path (str): Ruta a la carpeta que contiene 'german.data'.
                           Ejemplo: 'data/raw/statlog+german+credit+data'
        """
        self.data_path = data_path
        self.file_name = 'german.data'

    def load_and_prepare_data(self):
        """
        Carga el archivo, le asigna los nombres de columna correctos y
        lo devuelve como un DataFrame de pandas.
        """
        # Construye la ruta completa al archivo de forma segura
        file_path = os.path.join(self.data_path, self.file_name)

        # Nombres de las columnas según la documentación del dataset
        column_names = [
            'estado_cuenta_actual', 'duracion_credito_meses', 'historial_credito',
            'proposito_credito', 'monto_credito', 'ahorros_o_bonos',
            'tiempo_empleo', 'tasa_de_cuotas', 'estado_civil_y_sexo',
            'otros_deudores_o_garantes', 'tiempo_residencia_actual', 'propiedad',
            'edad', 'otros_planes_cuotas', 'vivienda', 'numero_creditos_existentes',
            'trabajo', 'numero_personas_dependientes', 'telefono',
            'trabajador_extranjero', 'clase'
        ]

        print(f"Cargando datos desde: {file_path}")
        
        try:
            # Leemos el archivo, que está separado por espacios en blanco
            df = pd.read_csv(file_path, sep=' ', header=None)
            # Asignamos los nombres de columna
            df.columns = column_names
            print("Datos cargados y columnas asignadas exitosamente.")
            return df
        
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta especificada: {file_path}")
            print("Por favor, asegúrate de que la ruta en el script principal sea correcta.")
            return None