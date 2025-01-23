
import pandas as pd

# Funciones usadas en analisis exploratorio

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
