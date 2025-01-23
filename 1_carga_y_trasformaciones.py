#Librerias necesarias 

import kagglehub as kh
import pandas as pd
import joblib
#Indice de codigo 

#0.Cargar datos
#1.Analisis de datos - books_data
#1.1Transformacion de nombres de columnas
#1.2Exploracion de tipo de datos
#1.3Analisis de datos nulos y transformaciones - books_data
#2.Transformaciones - books_data
#2.1.Reemplazo nulos por cero en ratingscount
#2.2.Transformacion published Date "se conserva solo el año"
#3.Base Books_data limpia
#4.Analisis de datos nulos - books_rating
#4.1.Analisis de duplicados - books_rating
#4.2.Analisis de datos nulos -books_rating
#4.3.Transformaciones - books_rating
#4.4.Transformaciones - books_rating
#5.Base Books_rating limpia

# Download latest version
import os

    
path = kh.dataset_download("mohamedbakhet/amazon-books-reviews")

#0.Cargar datos
df_brating = pd.read_csv('C:\\Users\\Usuario\\Desktop\\Prueba_tecnica_Bancolombia\\BOOKS__PROYECT\\Books_rating.csv')
df_bdata = pd.read_csv('C:\\Users\\Usuario\\Desktop\\Prueba_tecnica_Bancolombia\\BOOKS__PROYECT\\books_data.csv')

#1.Analisis de datos - books_data
#1.1Transformacion de nombres de columnas
df_bdata.columns = df_bdata.columns.str.lower()

#1.2Exploracion de tipo de datos
df_bdata.dtypes
df_bdata.duplicated().sum()



#1.3Analisis de datos nulos y transformaciones - books_data
#Analsis datos nulos

"""En este apartado se identifican varios datos nulos en diferentes variables, para su tratamiento se decidio
    -Eliminar los datos nulos de las variables 'Title', 'categories', 'authors', y 'publisheddate ya que son variables que no se pueden imputar
    -Reemplazar nulos de rating con cero ya que se asume que el libro no tiene rating
    -Se realizaron transformaciones de forma en las variables de nombre y categoria"""

df_bdata.isnull().sum()

df_bdata = df_bdata.dropna(subset=['title'])
df_bdata = df_bdata.dropna(subset=['categories'])
df_bdata = df_bdata.dropna(subset=['authors'])
df_bdata = df_bdata.dropna(subset=['publisheddate'])

#Se eliminan las columnas inecesarias para el modelo como imagen, previewlink e infolink
df_bdata = df_bdata.drop(['image', 'previewlink', 'infolink'], axis = 1)

#2.Transformaciones - books_data
#2.1.Reemplazo nulos por cero en ratingscount
df_bdata['ratingscount'] = df_bdata['ratingscount'].fillna(0)

#2.2.Transformacion publishedsate, categoria y autor 

#Transformacion año
df_bdata['publisheddate'] = df_bdata['publisheddate'].astype(str)
df_bdata['publisheddate'] = df_bdata['publisheddate'].apply(lambda x : x.strip())
df_bdata['publisheddate'] = df_bdata['publisheddate'].apply(lambda x : x[:4])

#Transformacion categoria
df_bdata['categories'] = df_bdata['categories'].astype(str)
df_bdata['categories'] = df_bdata['categories'].str.slice(1, -1)

#Transformacion autor
df_bdata['authors'] = df_bdata['authors'].astype(str)
df_bdata['authors'] = df_bdata['authors'].str.slice(2, -2)
df_bdata['authors'] = df_bdata['authors'].replace("'", "", regex=True)


# Convirtiendo todo a minúsculas
df_bdata = df_bdata.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

#3.Base Books_data limpia
df_clean_bdata = df_bdata.copy()

#4.Analisis de datos nulos - books_rating

df_brating.dtypes


#Convertimos variables texto a minuscula

#Convirtiendo todo a minusculas

df_brating.columns = df_brating.columns.str.lower()


#4.1.Analisis de duplicados - books_rating
"""En la exploracion de datos duplicados podemos observar que inicialmente 
    se encuentran 8mil datos duplicados en la base de datos, sin embargo, al revisar los datos duplicados
    se observa que son reviews distintos, por lo que se decide no eliminarlos"""
    
df_brating.duplicated().sum()
df_brating[df_brating.duplicated()] 

#4.2.Analisis de datos nulos -books_rating

"""Al analizae los datos nulos se observa que 
    - la variable price tiene gran parte de sus datos nulos, a pesar de esto se decide eliminar los nulos y conservar los datos sobrantes, pues este volumen de datos 
    es suficiente para el analisis y podriamos observar si la variable precio tiene algun tipo de relacion con las reseñas
    - eliminamos los librso que no tienen titulo , y que si los dejamos no podemos unir con la base de datos de books_data
    - Los campos userid y profilename tienen gran cantidad de datos nulos, estas pueden ser reseñas de personas anonimas o de personas
    que eliminaron sus cuentas, por lo que se reemplazan el nulo por anonimo"
    """

df_brating.isnull().sum()

#Se eliminan datos nulos
df_brating = df_brating.dropna(subset = ['price'])
df_brating = df_brating.dropna(subset=['title'])
df_brating = df_brating.dropna(subset=['review/summary'])


#4.4.Transformaciones - books_rating
#Reemplazamos campos de userid y profilename por anonimo
df_brating['user_id'] = df_brating['user_id'].fillna('anonimo')
df_brating['profilename'] = df_brating['profilename'].fillna('anonimo') 

#Transformacion fecha de publicacion de la reseña
df_brating['review/time'] = pd.to_datetime(df_brating['review/time'], unit='s') 
df_brating['review/time'] = df_brating['review/time'].astype(str)
df_brating['review/time'] = df_brating['review/time'].apply(lambda x : x.strip())
df_brating['review/time'] = df_brating['review/time'].apply(lambda x : x[:4])
df_brating = df_brating.rename(columns = {'review/time':'review_year'})

#Conversion a minusculas 

df_brating = df_brating.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)

#5.Base Books_rating limpia
df_clean_brating = df_brating.copy()


"""Debido al gran tamaño de la base de datos, tomamos una muestra de 300mil datos para conservar
el rendimiento del repositorio y no tener problemas en la actualizacion"""

df_brating_reducito_1 = df_clean_brating.sample(frac=0.4, random_state=42) 
df_brating_reducito_2 = df_clean_brating.sample(frac=0.4, random_state=455)


#6.Guardar bases de datos limpias
joblib.dump(df_clean_bdata, 'salidas\\df_clean_bdata.pkl',compress = 7)


joblib.dump(df_brating_reducito_1, 'salidas\\df_brating_reducito_1.pkl',compress= 8)
joblib.dump(df_brating_reducito_2, 'salidas\\df_brating_reducito_2.pkl',compress= 8)

#Confirmamos que el peso por archivo no sea mayor a 100 MB
peso = os.path.getsize('salidas\\df_brating_reducito_1.pkl')
peso_mb = peso/(1024*1024)