#Librerias utilizadas
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re   
import os



#1.Funciones
#2.Cargar datos
#3.Preparacion
#4.Transformacion
#5.Promedio de valoraciones por libro
#6.Analisis
#6.1.¿ Existe alguna relacion entre la valoracion score y la valoracion utilidad?
#6.2.Numero de reseñas y valoraciones totales
#6.3.Autores mas populares top(10)
#6.4.Categoria mas popular top(5)
#6.5.¿Como ha sido la evolucion de las reseñas a lo largo de los años?
#6.6.¿Como se distribuye el precio de los libros?
#7. Fragmentar y guardar base


#1.Funciones 

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

#2.Cargar datos

df_books_description = joblib.load('salidas\\df_clean_bdata.pkl')
df_books_rating_1 = joblib.load('salidas\\df_brating_reducito_1.pkl')
df_books_rating_2 = joblib.load('salidas\\df_brating_reducito_2.pkl')

#3.Preparacion

#Unimos las bases books_rating1 y books_rating2

df_books_rating = pd.concat([df_books_rating_1, df_books_rating_2])

#Cruzamos las bases books_description y books_rating

books_reviews = pd.merge(df_books_description, df_books_rating, on = 'title', how = 'inner')

"""Se agregaran columnas calculadas para realizar un analisis profundo 
de los datos"""

#Años transcurridos desde la publicacion del libro hasta la fecha de reseña

books_reviews['review_year'] = books_reviews['review_year'].astype(int)

#4.Transformacion

#Se encuentran valores defectuosos en la fecha de publicación, los cuales se proceden a eliminar

books_reviews = drop_special_reg(books_reviews, 'publisheddate')

books_reviews['publisheddate'] = books_reviews['publisheddate'].astype(int)

#Creamos la nueva columna 

books_reviews['years_since_published'] = np.abs(books_reviews['review_year'] - books_reviews['publisheddate'])

books_reviews['years_since_published'].describe()

#¿Hay alguna relacion entre years_since_published y review/score?

correlation_matrix = books_reviews[['years_since_published', 'review/score']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación entre Años y Review Score')
plt.show()

#5.promedio de valoraciones por libro 

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

#Eliminamos columnas que no se utilizaran
books_reviews_t = books_reviews_t.drop(['utilidad', 'total_personas'], axis = 1)

#6.Analisis

#6.1.¿ Existe alguna relacion entre la valoracion score y la valoracion utilidad?
#Grafico valoracion score vs valoracion utilidad

"""Se evidencia que no existe una correlacion entre estas variables lo cual quiere decir que 
 las reseñas con diferentes puntuaciones (1 a 5) son igualmente propensas a ser votadas como útiles o no útiles."""

plt.figure(figsize=(10, 6))
sns.scatterplot(x='review/score', y='review/helpfulness', data=books_reviews_t)
plt.title('Valoración Score vs Valoración Utilidad')
plt.xlabel('Valoración Score')
plt.ylabel('Valoración Utilidad')

#6.2.Numero de reseñas y valoraciones totales"""


reseñas = books_reviews_t['ratingscount'].sum()
scores = books_reviews_t['review/score'].count()

#6.3.Autores mas populares top(10)
autores_populares = books_reviews['authors'].value_counts().reset_index().head(10)
autores_populares.columns = ['authors', 'num_reviews']

plt.figure(figsize=(10, 6))
sns.barplot(x='num_reviews', y='authors', data=autores_populares)
plt.title('Autores más populares')
plt.xlabel('Número de reseñas')
plt.ylabel('Autores')

#6.4.Categoria mas popular top(5)
categoria_popular = books_reviews.groupby('categories')['review/text'].count().reset_index().sort_values(by = 'review/text', ascending = False).head(5)
categoria_popular.columns = ['categories', 'num_reviews']

plt.figure(figsize=(10, 6))
sns.barplot(x='num_reviews', y='categories', data=categoria_popular)
plt.title('Categorías más populares')
plt.xlabel('Número de reseñas')

#6.5.¿Como ha sido la evolucion de las reseñas a lo largo de los años?
# Grafico evolucion de reseñas por año
""""Se evidencia que el numero de reseñas ha disminuido a partir del año 2006"""
reseñas_por_año = books_reviews_t.groupby('review_year')['review/text'].count().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(x='review_year', y='review/text', data=reseñas_por_año)
plt.title('Evolución de reseñas por año')
plt.xlabel('Año')
plt.ylabel('Número de reseñas')

#6.6.¿Como se distribuye el precio de los libros?
#Grafico distribucion de precios - boxplot
books_reviews_t['price'].describe()

plt.figure(figsize=(10, 6))
sns.boxplot(x=books_reviews_t['price'])
plt.title('Distribución de precios')
plt.xlabel('Precio')

#Fragmentamos los datos para no saturar el repositorio
books_reviews_t.shape
books_reviews_t_1 = books_reviews_t.iloc[:150000, :]
books_reviews_t_2 = books_reviews_t.iloc[150000:320894, :]

# Guardar el DataFrame en un archivo pickle
joblib.dump(books_reviews_t_1, 'salidas\\cd_for_nlp_1.pkl', compress=9)
joblib.dump(books_reviews_t_2, 'salidas\\cd_for_nlp_2.pkl', compress=9)

# Confirmamos que el peso por archivo no sea mayor a 100 MB
peso = os.path.getsize('salidas\\cd_for_nlp_2.pkl')
peso_mb = peso/(1024*1024)