#Librerias utilizadas
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re   

#Funciones 


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

#Cargar datos

df_books_description = joblib.load('salidas\\df_clean_bdata.pkl')
df_books_rating_1 = joblib.load('salidas\\df_brating_reducito_1.pkl')
df_books_rating_2 = joblib.load('salidas\\df_brating_reducito_2.pkl')

#Unimos las bases books_rating1 y books_rating2

df_books_rating = pd.concat([df_books_rating_1, df_books_rating_2])

#Cruzamos las bases books_description y books_rating

books_reviews = pd.merge(df_books_description, df_books_rating, on = 'title', how = 'inner')

"""Se agregaran columnas calculadas para realizar un analisis profundo 
de los datos"""

#Años transcurridos desde la publicacion del libro hasta la fecha de reseña

books_reviews['review_year'] = books_reviews['review_year'].astype(int)

#Se encuentran valores defectuosos en la fecha de publicación, los cuales se proceden a eliminar


books_reviews = drop_special_reg(books_reviews, 'publisheddate')

books_reviews['publisheddate'] = books_reviews['publisheddate'].astype(int)

#Creamos la nueva columna 

books_reviews['years_since_published'] = np.abs(books_reviews['review_year'] - books_reviews['publisheddate'])

#Calculamos el promedio de valoraciones por libro 

""" 
- Se uso la funcion transformar para separar la variable review/helpfulness en dos columnas con el fin de calcular el promedio de valoraciones
y adaptar los datos, con ello obtendremos un puntaje en el cual, entre mas cercano sea a uno, mayor ayuda brindo la reseña
a los usuarios.

- Se uso una funcion para calcular el promedio de los scores y la utilidad de tal forma que se obtuvo un promedio de valoraciones por libro
en el cual se puede observar que libros tienen mejor valoracion y cuales son mas utiles para los usuarios

"""

books_reviews_t = transformacion_review_helpfulness(books_reviews)

valoracion_score = calculo_valoracion_promedio_por_libro(books_reviews_t)[1].sort_values(by = 'review/score', ascending = False)
valoracion_utilidad = calculo_valoracion_promedio_por_libro(books_reviews_t)[0].sort_values(by = 'review/helpfulness', ascending = False)

#¿ Existe alguna relacion entre la valoracion score y la valoracion utilidad?
#Grafico valoracion score vs valoracion utilidad

"""Se evidencia que no existe una correlacion entre estas variables lo cual quiere decir que 
 las reseñas con diferentes puntuaciones (1 a 5) son igualmente propensas a ser votadas como útiles o no útiles."""

plt.figure(figsize=(10, 6))
sns.scatterplot(x='review/score', y='review/helpfulness', data=books_reviews_t)
plt.title('Valoración Score vs Valoración Utilidad')
plt.xlabel('Valoración Score')
plt.ylabel('Valoración Utilidad')

#¿como afecta el precio en el score de los libros?
#Grafico precio vs valoracion score

plt.figure(figsize=(10, 6))
sns.lineplot(x='price', y='review/score', data=books_reviews_t)
plt.title('Precio vs Valoración Score')
plt.xlabel('Precio')
plt.ylabel('Valoración Score')
