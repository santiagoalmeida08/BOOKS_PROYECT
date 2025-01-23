#Librerias
import os
import pandas as pd
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Cargamos el dataset y lo unimos

def carga_dataset_nlp(path_1,path_2): 
    df1 = joblib.load(path_1)
    df2 = joblib.load(path_2)
    df_nlp = pd.concat([df1, df2])
    return df_nlp

# Función para limpiar el texto
def limpiar_texto(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    # Eliminar espacios adicionales
    texto = texto.strip()
    return texto

# Función para obtener el puntaje de sentimiento
def obtener_sentimiento(texto):
    return analyzer.polarity_scores(texto)['compound'] 

def sentimiento_promedio_por_libro(df, columna_libro):
    return df.groupby(columna_libro)['sentimiento'].mean().reset_index()

def sentimiento_promedio_por_categoria(df, columna_categoria):    
    return df.groupby(columna_categoria)['sentimiento'].mean().reset_index()


#2.Carga data set

df_nlp = carga_dataset_nlp('salidas\\cd_for_nlp_1.pkl','salidas\\cd_for_nlp_2.pkl')

#Cambiamos el tipo de datos para limpiar

df_nlp.dtypes
df_nlp['review/text'] = df_nlp['review/text'].astype(str)
df_nlp['review/summary'] = df_nlp['review/summary'].astype(str)


#Limpiza del dataset 

df_nlp['review/text'] = df_nlp['review/text'].apply(limpiar_texto)

# Inicializar el analizador de sentimientos VADER
analyzer = SentimentIntensityAnalyzer()


# Aplicar el análisis de sentimientos al dataset
df_nlp['sentimiento'] = df_nlp['review/text'].apply(obtener_sentimiento)

#Clasificacion sentimientos segun la escala del compund

df_nlp['des_sentimiento'] = df_nlp['sentimiento'].apply(lambda x: 'reseña positiva' if x > 0.05 else ('reseña negativa' if x < -0.05 else 'reseña neutral'))


# Contar la cantidad de cada tipo de sentimiento
sentimiento_counts = df_nlp['des_sentimiento'].value_counts()

# Grafico de pastel
plt.figure(figsize=(8, 8))
plt.pie(sentimiento_counts, labels=sentimiento_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Percepción General Basada en Sentimientos')
plt.axis('equal')  # Para asegurar que el gráfico sea un círculo
plt.show()

print(f'----------------------------------------------------')

print('Resumen de sentimientos a nivel general:')
print(f'Porcentaje de reseñas positivas: {sentimiento_counts["Reseña Positiva"] / len(df_nlp) * 100:.2f}%')
print(f'Porcentaje de reseñas negativas: {sentimiento_counts["Reseña Negativa"] / len(df_nlp) * 100:.2f}%')
print(f'Porcentaje de reseñas neutrales: {sentimiento_counts["Reseña Neutral"] / len(df_nlp) * 100:.2f}%')

# Calcular el sentimiento promedio por libro
df_sentimiento_libro = sentimiento_promedio_por_libro(df_nlp, 'title')
df_sentimiento_libro.describe()

# Calcular el sentimiento promedio por categoría
df_sentimiento_categoria = sentimiento_promedio_por_categoria(df_nlp, 'categories').sort_values(by='sentimiento', ascending=True)

#Guardar base.
#Fragmentamos los datos para no saturar el repositorio
df_nlp.shape
df_nlp_1 = df_nlp.iloc[:150000, :]
df_nlp_2 = df_nlp.iloc[150000:320894, :]

# Guardar el DataFrame en un archivo pickle
joblib.dump(df_nlp_1, 'salidas\\df_npl_1.pkl', compress=9)
joblib.dump(df_nlp_2, 'salidas\\df_nlp_2.pkl', compress=9)

# Confirmamos que el peso por archivo no sea mayor a 100 MB
peso = os.path.getsize('salidas\\df_nlp_2.pkl')
peso_mb = peso/(1024*1024)







