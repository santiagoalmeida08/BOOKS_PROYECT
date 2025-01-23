import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Funciones

def carga_dataset_nlp(path_1, path_2): 
    df1 = joblib.load(path_1)
    df2 = joblib.load(path_2)
    df_nlp = pd.concat([df1, df2])
    return df_nlp

# Cargar datos
df = carga_dataset_nlp('salidas\\df_npl_1.pkl', 'salidas\\df_nlp_2.pkl')

# Calcular el número de reseñas por libro
conteo_reseñas = df['title'].value_counts().reset_index()
conteo_reseñas.columns = ['title', 'num_reseñas']

# Calcular el promedio de puntaje por libro
promedio_puntaje = df.groupby('title')['review/score'].mean().reset_index()
promedio_puntaje.columns = ['title', 'promedio_puntaje']

# Calcular el sentimiento promedio por libro
sentimiento_promedio = df.groupby('title')['sentimiento'].mean().reset_index()
sentimiento_promedio.columns = ['title', 'sentimiento_promedio']

# Unir todas las métricas en un solo DataFrame
df_libros = pd.merge(conteo_reseñas, promedio_puntaje, on='title')
df_libros = pd.merge(df_libros, sentimiento_promedio, on='title')

# Normalizar las métricas
scaler = MinMaxScaler()
df_libros[['num_reseñas', 'promedio_puntaje', 'sentimiento_promedio']] = scaler.fit_transform(df_libros[['num_reseñas', 'promedio_puntaje', 'sentimiento_promedio']])

# Combinar las métricas en un puntaje único
df_libros['puntaje_combinado'] = df_libros['num_reseñas'] + df_libros['promedio_puntaje'] + df_libros['sentimiento_promedio']

# Obtener los top 10 libros basándose en el puntaje combinado
top_10_libros = df_libros.sort_values(by='puntaje_combinado', ascending=False).head(10)

# Mostrar los resultados
print("Top 10 libros recomendados:")
print(top_10_libros[['title','puntaje_combinado']])