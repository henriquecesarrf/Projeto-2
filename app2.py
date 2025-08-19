import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px

st.set_page_config(page_title="Previsão de Demanda", layout="wide")

st.title("📈 DemandAI")
st.markdown("Envie um arquivo com histórico de vendas.")

# Upload do arquivo
uploaded_file = st.file_uploader("Carregue seu arquivo CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Ler arquivo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Exibir preview
        st.subheader("📊 Dados Carregados")
        st.dataframe(df.head())

        # Validar colunas
        colunas_necessarias = {"Produto", "Ano-Mês", "Filial", "Quantidade de Vendas"}
        if not colunas_necessarias.issubset(df.columns):
            st.error(f"O arquivo deve conter as colunas: {', '.join(colunas_necessarias)}")
        else:
            # Criar filtros
            produto_escolhido = st.selectbox("Escolha o produto", sorted(df["Produto"].unique()))
            filial_escolhida = st.selectbox("Escolha a filial", sorted(df["Filial"].unique()))

            # Filtrar dados
            df_filtrado = df[
                (df["Produto"] == produto_escolhido) &
                (df["Filial"] == filial_escolhida)
            ].copy()

            # Converter Ano-Mês para datetime
            df_filtrado["ds"] = pd.to_datetime(df_filtrado["Ano-Mês"], format="%Y-%m")
            df_filtrado["y"] = df_filtrado["Quantidade de Vendas"]

            # Ordenar por data
            df_filtrado = df_filtrado[["ds", "y"]].sort_values("ds")

            # Verificar se tem dados suficientes
            if len(df_filtrado) < 3:
                st.warning("Poucos dados para prever. Adicione mais histórico.")
            else:
                # Criar modelo Prophet
                modelo = Prophet()
                modelo.fit(df_filtrado)

                # Criar datas futuras
                futuro = modelo.make_future_dataframe(periods=6, freq="M")  # 6 meses
                previsao = modelo.predict(futuro)

                # Gráfico
                fig = px.line(previsao, x="ds", y="yhat", title=f"Previsão de Vendas - {produto_escolhido} ({filial_escolhida})")
                fig.add_scatter(x=df_filtrado["ds"], y=df_filtrado["y"], mode="markers", name="Histórico")
                st.plotly_chart(fig, use_container_width=True)

                # Tabela de previsão
                st.subheader("📅 Previsão")
                st.dataframe(previsao[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6))

    except Exception as e:
        st.error(f"Erro ao processar: {e}")