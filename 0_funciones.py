
#Librerias

import pandas as pd
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



#1. Funciones usadas en analisis exploratorio
#2. Funciones usadas en analisis de sentimientos


#1. Funciones usadas en analisis exploratorio

def drop_special_reg(df, columna):
    """
    Elimina registros en un DataFrame que contienen caracteres especiales
    o tienen menos de 4 caracteres en una columna específica.
    
    :param df: DataFrame que contiene los datos.
    :param columna: Nombre de la columna a analizar.
    :return: DataFrame sin los registros que cumplen las condiciones.
    """
    # Patrón para caracteres especiales (todo excepto letras y números)
    patron = r'[^a-zA-Z0-9]'
    
    # Filtrar los registros que NO cumplen las condiciones
    df_limpio = df[
        (df[columna].str.len() >= 4) &  # Longitud mayor o igual a 4
        (~df[columna].str.contains(patron, regex=True))  # No contiene caracteres especiales
    ]
    
    return df_limpio


def transformacion_review_helpfulness(df):
    """
    Transforma la columna review/helpfulness en dos columnas y calcula el promedio de valoraciones.
    
    :param df: DataFrame que contiene los datos.
    :return: DataFrame con las columnas transformadas y el promedio de valoraciones.
    """
    # Separamos la columna review/helpfulness en dos columnas
    df[['utilidad', 'total_personas']] = df['review/helpfulness'].str.split('/', expand=True)
    
    # Calculamos el promedio de valoraciones
    df['review/helpfulness'] = df['utilidad'].astype(float) / df['total_personas'].astype(float)
    df['review/helpfulness'] = df['review/helpfulness'].fillna(0)
    
    return df


def calculo_valoracion_promedio_por_libro(df):
    """
    Calcula el promedio de valoraciones por libro.
    
    :param df: DataFrame que contiene los datos.
    :return: DataFrame con el promedio de valoraciones por libro.
    """
    valoracion_promedio_utilidad_reseña = df.groupby('title')['review/helpfulness'].mean().reset_index()
    valoracion_promedio_score = df.groupby('title')['review/score'].mean().reset_index()
    return valoracion_promedio_utilidad_reseña, valoracion_promedio_score

#2.Funciones Analisis de Sentimientos


def carga_dataset_nlp(path_1,path_2): 

    """
    Carga y combina dos datasets NLP desde las rutas especificadas.
    Args:
        path_1 (str): Ruta al primer archivo de dataset.
        path_2 (str): Ruta al segundo archivo de dataset.
    Returns:
        DataFrame: Un DataFrame que contiene la combinación de los dos datasets cargados.
    """
    
    df1 = joblib.load(path_1)
    df2 = joblib.load(path_2)
    df_nlp = pd.concat([df1, df2])
    return df_nlp


def limpiar_texto(texto):
    """
    Limpia el texto dado aplicando las siguientes transformaciones:
    1. Convierte el texto a minúsculas.
    2. Elimina caracteres especiales y números, dejando solo letras y espacios.
    3. Elimina espacios adicionales al inicio y al final del texto.
    Args:
        texto (str): El texto a limpiar.
    Returns:
        str: El texto limpio.
    """
    
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    # Eliminar espacios adicionales
    texto = texto.strip()
    return texto


def obtener_sentimiento(texto):
    """
    Analiza el sentimiento de un texto dado y devuelve la puntuación compuesta.
    Args:
        texto (str): El texto a analizar.
    Returns:
        float: La puntuación compuesta del sentimiento del texto.
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(texto)['compound'] 

def sentimiento_promedio_por_libro(df, columna_libro):    
    return df.groupby(columna_libro)['sentimiento'].mean().reset_index()

def sentimiento_promedio_por_categoria(df, columna_categoria):    
    return df.groupby(columna_categoria)['sentimiento'].mean().reset_index()