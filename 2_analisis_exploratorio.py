#Librerias utilizadas
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Cargar datos

df_books = joblib.load('salidas\\df_clean_bdata.pkl')
df_books_rating = joblib.load('salidas\\df_brating_reducito_1.pkl')