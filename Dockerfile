# Paso 1: Usar una imagen base oficial de Python
# Usamos la versión 'slim' para que nuestra imagen final sea más ligera.
FROM python:3.11-slim

# Paso 2: Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Paso 3: Copiar el archivo de dependencias
# Se copia primero para aprovechar el caché de Docker y acelerar construcciones futuras.
COPY requirements.txt .

# Paso 4: Instalar las dependencias
# --no-cache-dir asegura que no se guarde el caché de pip, manteniendo la imagen pequeña.
RUN pip install --no-cache-dir -r requirements.txt

# Paso 5: Copiar todo el código de nuestro proyecto al contenedor
COPY src/ ./src
COPY prefect_flow.py .
COPY data/ ./data

# Paso 6: Comando por defecto que se ejecutará al iniciar el contenedor
CMD ["python", "prefect_flow.py"]