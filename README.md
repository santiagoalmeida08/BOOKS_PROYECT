# Proyecto de Análisis de Reseñas de Libros

## Propósito del Proyecto

Este proyecto tiene como objetivo analizar las reseñas de libros para identificar patrones y tendencias. Utilizamos diversas técnicas de análisis de datos para explorar la relación entre las valoraciones de los libros, los autores más populares, la evolución de las reseñas a lo largo del tiempo, y otros aspectos relevantes.

Adicionalmente se explora el analisis de sentimientos evaluando las reseñas de los lectores mediante tecnicas de NLP.

## Instrucciones para Ejecutar el Análisis

1. Abrir la carpeta del proyecto 

2. Activar el entorno virtual, sigue los siguientes pasos:

    En caso de que no tengas un entorno virtual creado ejecuta el siguiente comando
    ```bash
    python -m venv env #instalacion entorno 
    pip list #verifica que tengas los requerimientos del proyecto instalados
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser #Permisos powershell
    env\\Scripts\\Activate.ps1 #Activar entorno

3. Instalar las dependencias: Asegúrate de tener pip instalado y ejecuta:

    ```bash
    pip install -r requirements.txt

4. Ejecutar el análisis: Ejecuta los scripts de análisis en el siguiente orden:
    
    ```bash
    python 1_carga_y_trasformaciones.py
    python 2_analisis_exploratorio.py


