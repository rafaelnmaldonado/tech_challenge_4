import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

DATA_FINAL_TREINO = '11-01-2023'
indice = "BZ=F"
inicio = "2009-01-01"
dados_acao = yf.download(indice, inicio)
df_cotacoes = pd.DataFrame({indice: dados_acao['Close']})
print(df_cotacoes)
df = df_cotacoes
df.index.name = 'ds'
df.rename(columns={'BZ=F': 'y'}, inplace=True)
treino = df.loc[df.index < DATA_FINAL_TREINO]
teste = df.loc[df.index >= DATA_FINAL_TREINO]

fig, ax = plt.subplots(figsize=(18,6))
treino.plot(ax=ax, label='Conjunto de treinamento', title='Dados de treino e teste')
teste.plot(ax=ax, label='Conjunto de teste')
ax.legend(['Conjunto de treinamento', 'Conjunto de teste'])

def adiciona_periodos(df):
    df = df.copy()
    df['dia_do_ano'] = df.index.dayofyear
    df['dia_do_mes'] = df.index.day
    df['dia_da_semana'] = df.index.dayofweek
    df['trimestre'] = df.index.quarter
    df['mes'] = df.index.month
    df['ano'] = df.index.year
    df['semana_do_ano'] = df.index.isocalendar().week
    return df

df = adiciona_periodos(df)
treino = adiciona_periodos(treino)
teste = adiciona_periodos(teste)

PERIODOS = ['dia_do_ano', 'dia_do_mes', 'dia_da_semana', 'trimestre', 'mes', 'ano', 'semana_do_ano']
Y = 'y'

X_treino = treino[PERIODOS]
Y_treino = treino[Y]

X_teste = teste[PERIODOS]
Y_teste = teste[Y]

reg = xgb.XGBRegressor(base_score=0.5,
                      booster='gbtree',
                      objective='reg:tweedie',
                      max_depth=3,
                      learning_rate=1.5)
reg.fit(X_treino, Y_treino,
        eval_set=[(X_treino, Y_treino), (X_teste, Y_teste)],
        verbose=100)

relevancia_periodos = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['relevancia'])
# relevancia_periodos.sort_values('relevancia', ascending=False, inplace=True)

st.title('Tech challenge 4')
st.line_chart(df_cotacoes)
st.pyplot(fig)
st.bar_chart(relevancia_periodos)

teste['resultado'] = reg.predict(X_teste)
df = df.merge(teste[['resultado']], how='left', left_index=True, right_index=True)
ax = df['y'].plot(figsize=(18, 6))
df['resultado'].plot(ax=ax, style='-')
plt.legend(['Dados', 'Previsões'])
ax.set_title('Petróleo')

mape = mean_absolute_percentage_error(df.loc[df.index > DATA_FINAL_TREINO]['y'], df.loc[df.index > DATA_FINAL_TREINO]['resultado'])

st.pyplot(fig)
st.text(f'MAPE: {mape*100:.3f}%')