import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

DATA_FINAL_TREINO = '09-01-2023'
indice = "BZ=F"
inicio = "2009-01-01"
dados_acao = yf.download(indice, inicio)
df_cotacoes = pd.DataFrame({indice: dados_acao['Close']})
print(df_cotacoes)
df = df_cotacoes
df.rename(columns={'Date': 'ds', 'BZ=F': 'y'}, inplace=True)
treino = df.loc[df.index < DATA_FINAL_TREINO]
teste = df.loc[df.index >= DATA_FINAL_TREINO]

fig, ax = plt.subplots(figsize=(18,6))
treino.plot(ax=ax, label='Conjunto de treinamento', title='Dados de treino e teste')
teste.plot(ax=ax, label='Conjunto de teste')
ax.legend(['Conjunto de treinamento', 'Conjunto de teste'])

st.title('Tech challenge 4')
st.line_chart(df_cotacoes)
st.pyplot(fig)