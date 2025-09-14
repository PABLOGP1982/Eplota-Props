import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.markdown("""
    <style>
    .scrollable-table {
        overflow-x: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

def map_sqx_csv_to_standard(df):
    rename_dict = {
        "Open time": "Open Time",
        "Close time": "Close Time",
        "Open price": "Price",
        "Close price": "Close Price",
        "Symbol": "Item",
        "Profit/Loss": "Profit",
    }
    columns_renamed = {k: v for k, v in rename_dict.items() if k in df.columns}
    df = df.rename(columns=columns_renamed)
    return df

def simular_challenge(trades_full, month_start, params):
    CAPITAL_FASE = 100_000
    trades = trades_full[trades_full['Close Time'] >= month_start].copy()
    if trades.empty:
        return None

    resultado = {
        'Inicio': month_start,
        'Duracion F1': 0,
        'Duracion F2': 0,
        'Duracion Funded': 0,
        'Duracion F1 días': 0,       # <-- NUEVO
        'Duracion F2 días': 0,
        'Duracion Funded días': 0,
        'Estado': None,
        'Motivo Cierre': None,
        'Profit Final': 0.0,
        'Trades totales': 0,
        'Duracion Total': 0,
        'Trades usados detallados': [],
        'Fase inicio idx': {'F1': None, 'F2': None, 'Funded': None}
    }

    estado = "Fase1"
    capital = CAPITAL_FASE
    objetivo_f1 = CAPITAL_FASE * (1 + params['pct_fase1'] / 100)
    max_dd = CAPITAL_FASE * params['drawdown_max'] / 100
    max_dd_diario = CAPITAL_FASE * params['drawdown_diario'] / 100
    duration_f1, duration_f2, duration_funded = 0, 0, 0

    # Guardar fechas de inicio y fin de fases:
    f1_first, f1_last = None, None
    f2_first, f2_last = None, None
    fund_first, fund_last = None, None

    dia_actual = None
    dd_dia_actual = 0
    acum_profit_funded = 0
    watermark_funded = 0
    funded_started = False

    for idx, row in trades.iterrows():
        fecha_trade = row['Close Time']

        if estado == "Fase1":
            if f1_first is None:
                f1_first = fecha_trade
            f1_last = fecha_trade
            profit_trade = row['Profit'] * params['riesgo_fase1']
            current_fase = "Fase1"
        elif estado == "Fase2":
            if f2_first is None:
                f2_first = fecha_trade
            f2_last = fecha_trade
            profit_trade = row['Profit'] * params['riesgo_fase2']
            current_fase = "Fase2"
        elif estado == "Funded":
            if fund_first is None:
                fund_first = fecha_trade
            fund_last = fecha_trade
            profit_trade = row['Profit'] * params['riesgo_fondeado']
            current_fase = "Funded"
        else:
            profit_trade = 0
            current_fase = ""

        resultado['Trades usados detallados'].append({
            **row.to_dict(),
            "Profit (riesgo fase)": profit_trade,
            "Fase": current_fase,
            "Capital": capital,
            "MaxDD": max_dd,
            "Cleaned_EA": row.get("Cleaned_EA", "")
        })

        nueva_fecha = row['Close Time'].date()
        if dia_actual != nueva_fecha:
            dia_actual = nueva_fecha
            dd_dia_actual = 0
        dd_dia_actual += -profit_trade if profit_trade < 0 else 0

        capital += profit_trade

        if estado == "Fase1":
            duration_f1 += 1
            if dd_dia_actual < -max_dd_diario:
                # durataion en días:
                if f1_first and f1_last:
                    resultado['Duracion F1 días'] = (f1_last.date() - f1_first.date()).days + 1
                resultado['Estado'] = "Fallido"
                resultado['Motivo Cierre'] = "DD Diario F1"
                resultado['Duracion F1'] = duration_f1
                resultado['Profit Final'] = 0
                resultado['Trades totales'] = duration_f1
                resultado['Duracion Total'] = duration_f1
                return resultado
            if capital <= CAPITAL_FASE - max_dd:
                if f1_first and f1_last:
                    resultado['Duracion F1 días'] = (f1_last.date() - f1_first.date()).days + 1
                resultado['Estado'] = "Fallido"
                resultado['Motivo Cierre'] = "DD Máximo F1"
                resultado['Duracion F1'] = duration_f1
                resultado['Profit Final'] = 0
                resultado['Trades totales'] = duration_f1
                resultado['Duracion Total'] = duration_f1
                return resultado
            if capital >= objetivo_f1:
                if f1_first and f1_last:
                    resultado['Duracion F1 días'] = (f1_last.date() - f1_first.date()).days + 1
                estado = "Fase2"
                capital = CAPITAL_FASE
                objetivo_f2 = CAPITAL_FASE * (1 + params['pct_fase2'] / 100)
                max_dd = CAPITAL_FASE * params['drawdown_max'] / 100
                max_dd_diario = CAPITAL_FASE * params['drawdown_diario'] / 100
                duration_f2 = 0
                dia_actual = None
                dd_dia_actual = 0
                continue
        elif estado == "Fase2":
            duration_f2 += 1
            if dd_dia_actual < -max_dd_diario:
                if f2_first and f2_last:
                    resultado['Duracion F2 días'] = (f2_last.date() - f2_first.date()).days + 1
                resultado['Estado'] = "Fallido"
                resultado['Motivo Cierre'] = "DD Diario F2"
                resultado['Duracion F1'] = duration_f1
                resultado['Duracion F2'] = duration_f2
                resultado['Profit Final'] = 0
                resultado['Trades totales'] = duration_f1 + duration_f2
                resultado['Duracion Total'] = duration_f1 + duration_f2
                return resultado
            if capital <= CAPITAL_FASE - max_dd:
                if f2_first and f2_last:
                    resultado['Duracion F2 días'] = (f2_last.date() - f2_first.date()).days + 1
                resultado['Estado'] = "Fallido"
                resultado['Motivo Cierre'] = "DD Máximo F2"
                resultado['Duracion F1'] = duration_f1
                resultado['Duracion F2'] = duration_f2
                resultado['Profit Final'] = 0
                resultado['Trades totales'] = duration_f1 + duration_f2
                resultado['Duracion Total'] = duration_f1 + duration_f2
                return resultado
            if capital >= objetivo_f2:
                if f2_first and f2_last:
                    resultado['Duracion F2 días'] = (f2_last.date() - f2_first.date()).days + 1
                estado = "Funded"
                capital = CAPITAL_FASE
                max_dd = CAPITAL_FASE * params['drawdown_max'] / 100
                max_dd_diario = CAPITAL_FASE * params['drawdown_diario'] / 100
                duration_funded = 0
                dia_actual = None
                dd_dia_actual = 0
                funded_started = True
                continue
        elif estado == "Funded":
            duration_funded += 1
            if fund_first is None:
                fund_first = fecha_trade
            fund_last = fecha_trade
            acum_profit_funded += profit_trade
            if acum_profit_funded > watermark_funded:
                watermark_funded = acum_profit_funded
            if dd_dia_actual < -max_dd_diario:
                if fund_first and fund_last:
                    resultado['Duracion Funded días'] = (fund_last.date() - fund_first.date()).days + 1
                resultado['Estado'] = "Funded quemada"
                resultado['Motivo Cierre'] = "DD Diario Funded"
                resultado['Duracion F1'] = duration_f1
                resultado['Duracion F2'] = duration_f2
                resultado['Duracion Funded'] = duration_funded
                resultado['Profit Final'] = round(watermark_funded, 0)
                resultado['Trades totales'] = duration_f1 + duration_f2 + duration_funded
                resultado['Duracion Total'] = duration_f1 + duration_f2 + duration_funded
                return resultado
            if funded_started:
                if acum_profit_funded <= watermark_funded - max_dd:
                    if fund_first and fund_last:
                        resultado['Duracion Funded días'] = (fund_last.date() - fund_first.date()).days + 1
                    resultado['Estado'] = "Funded quemada"
                    resultado['Motivo Cierre'] = "DD Máximo Funded"
                    resultado['Duracion F1'] = duration_f1
                    resultado['Duracion F2'] = duration_f2
                    resultado['Duracion Funded'] = duration_funded
                    resultado['Profit Final'] = round(watermark_funded, 0)
                    resultado['Trades totales'] = duration_f1 + duration_f2 + duration_funded
                    resultado['Duracion Total'] = duration_f1 + duration_f2 + duration_funded
                    return resultado

    if estado == "Funded":
        if fund_first and fund_last:
            resultado['Duracion Funded días'] = (fund_last.date() - fund_first.date()).days + 1
        resultado['Estado'] = "Funded (vivo)"
        resultado['Motivo Cierre'] = "Acabaron trades"
        resultado['Duracion F1'] = duration_f1
        resultado['Duracion F2'] = duration_f2
        resultado['Duracion Funded'] = duration_funded
        resultado['Profit Final'] = round(watermark_funded, 0)
        resultado['Trades totales'] = duration_f1 + duration_f2 + duration_funded
        resultado['Duracion Total'] = duration_f1 + duration_f2 + duration_funded
        return resultado
    if estado == "Fase2":
        if f2_first and f2_last:
            resultado['Duracion F2 días'] = (f2_last.date() - f2_first.date()).days + 1
        resultado['Estado'] = "No completado"
        resultado['Motivo Cierre'] = "No cumple objetivo F2"
        resultado['Duracion F1'] = duration_f1
        resultado['Duracion F2'] = duration_f2
        resultado['Profit Final'] = 0
        resultado['Trades totales'] = duration_f1 + duration_f2
        resultado['Duracion Total'] = duration_f1 + duration_f2
        return resultado
    if f1_first and f1_last:
        resultado['Duracion F1 días'] = (f1_last.date() - f1_first.date()).days + 1
    resultado['Estado'] = "No completado"
    resultado['Motivo Cierre'] = "No cumple objetivo F1"
    resultado['Duracion F1'] = duration_f1
    resultado['Profit Final'] = 0
    resultado['Trades totales'] = duration_f1
    resultado['Duracion Total'] = duration_f1
    return resultado

# ====== APP MAIN ======

st.set_page_config(page_title="ExploTA Props", layout="wide")
st.title("ExploTA Props")

with st.expander("Ayuda - ¿Cómo funciona cada parámetro?"):
    st.markdown("""
    - **Fase 1 (% para pasar)**: Porcentaje de beneficio para pasar Fase 1.
    - **Fase 2 (% para pasar)**: Porcentaje de beneficio para pasar Fase 2.
    - **Drawdown Diario/Máx**: Pérdidas máximas permitidas.
    - **Riesgo**: Multiplicador de cada fase.
    """)

uploaded_file = st.file_uploader("Cargar archivo CSV de trades (exportado de SQX)", type=['csv'])
col1, col2, col3, col4 = st.columns(4)
with col1:
    pct_fase1 = st.number_input("Fase 1 (% profit)", 0.0, 50.0, 8.0)
    riesgo_fase1 = st.number_input("Riesgo Fase 1 (x)", 0.1, 100.0, 10.0)
with col2:
    pct_fase2 = st.number_input("Fase 2 (% profit)", 0.0, 50.0, 5.0)
    riesgo_fase2 = st.number_input("Riesgo Fase 2 (x)", 0.1, 100.0, 10.0)
with col3:
    drawdown_diario = st.number_input("Drawdown Diario (%)", 0.0, 50.0, 5.0)
    riesgo_fondeado = st.number_input("Riesgo Fondeado (x)", 0.1, 100.0, 10.0)
with col4:
    drawdown_max = st.number_input("Drawdown Máximo (%)", 0.0, 50.0, 10.0)
    reset_btn = st.button("🔄 Reset")
if reset_btn:
    st.experimental_rerun()

procesar = st.button("Procesar desafíos")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = df.columns.str.strip()
    df = map_sqx_csv_to_standard(df)
    df['Close Time'] = pd.to_datetime(df['Close Time'])

    df = df.sort_values('Close Time')
    df['Month'] = df['Close Time'].dt.to_period('M')
    meses = sorted(df['Month'].unique())

    with st.expander("Resumen de desafíos", expanded=False):  # <--- NUEVO expander
        if procesar:
            params = {
                'pct_fase1': pct_fase1,
                'pct_fase2': pct_fase2,
                'drawdown_diario': drawdown_diario,
                'drawdown_max': drawdown_max,
                'riesgo_fase1': riesgo_fase1,
                'riesgo_fase2': riesgo_fase2,
                'riesgo_fondeado': riesgo_fondeado
            }
            resultados = []
            for periodo in meses:
                mes_ini = pd.Timestamp(periodo.start_time)
                mask = (df['Close Time'] >= mes_ini) & (df['Close Time'] < (mes_ini + pd.offsets.MonthEnd(1)))
                trades_mes = df[mask]
                if not trades_mes.empty:
                    res = simular_challenge(df, mes_ini, params)
                    if res:
                        resultados.append(res)
            if resultados:
                res_df = pd.DataFrame(resultados)
                res_df['Inicio'] = pd.to_datetime(res_df['Inicio']).dt.strftime('%Y-%m-%d')
                res_df = res_df.sort_values('Inicio')
                res_df["Profit Final"] = res_df["Profit Final"].round(0).astype(int)

                with st.expander("Filtros"):
                    colf1, colf2, colf3 = st.columns(3)
                    states = res_df['Estado'].unique().tolist()
                    estado_filtro = colf1.multiselect("Estado:", states, default=states)
                    min_profit = colf2.number_input(
                        "Profit Final mínimo", int(res_df["Profit Final"].min()),
                        int(res_df["Profit Final"].max()), 0)
                    fechas = res_df["Inicio"].unique()
                    fecha_ini = colf3.selectbox("Desde (Inicio)", fechas, index=0)
                    fecha_fin = colf3.selectbox("Hasta (Inicio)", fechas, index=len(fechas) - 1)

                filtered = res_df[
                    (res_df['Estado'].isin(estado_filtro)) &
                    (res_df["Profit Final"] >= min_profit) &
                    (res_df["Inicio"] >= fecha_ini) &
                    (res_df["Inicio"] <= fecha_fin)
                ]

                with st.container():
                    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
                    kpi_total = len(filtered)
                    kpi_success = (filtered["Estado"].isin(["Funded quemada", "Funded (vivo)"])).sum()
                    kpi_success_pct = 100 * kpi_success / kpi_total if kpi_total else 0
                    kpi_avg_watermark = filtered.loc[filtered['Profit Final'] > 0, 'Profit Final'].mean() if kpi_success else 0
                    kpi_avg_duration = filtered.loc[filtered['Profit Final'] > 0, 'Duracion Total'].mean() if kpi_success else 0
                    col_kpi1.metric("Total desafíos", kpi_total)
                    col_kpi2.metric("Fondeados (superados)", f"{kpi_success} ({kpi_success_pct:.1f}%)")
                    col_kpi3.metric("Profit watermark medio", f"{int(round(kpi_avg_watermark))}")
                    col_kpi4.metric("Media días hasta perder/cobrar", f"{kpi_avg_duration:.1f}")

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Media duración F1", f"{filtered['Duracion F1'].mean():.1f}")
                    col_m2.metric("Media duración F2", f"{filtered['Duracion F2'].mean():.1f}")
                    col_m3.metric("Media duración Fondeado", f"{filtered['Duracion Funded'].mean():.1f}")
                    col_m4.metric("Media Profit Final", f"{filtered['Profit Final'].mean():.1f}")

                excelout = io.BytesIO()
                filtered.to_excel(excelout, index=False)
                excelout.seek(0)
                st.download_button(
                    "Descargar Excel Resumen",
                    data=excelout,
                    file_name="challenges_resumen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                with st.expander("Mostrar gráficos", expanded=False):
                    st.markdown("#### 📊 Gráficos")
                    if not filtered.empty:
                        fig1 = px.bar(filtered, x="Inicio", y="Profit Final", color="Estado", title="Resultado de cada desafío", height=300)
                        st.plotly_chart(fig1, use_container_width=True)
                        fig2 = px.pie(filtered, names="Motivo Cierre", title="Motivo de cierre (%)")
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No hay datos para graficar con estos filtros.")

                with st.expander("Mostrar tabla principal", expanded=False):
                    cols_resumidas = [c for c in filtered.columns if 'Trade' not in c and 'Fase' not in c or c == "Inicio"]
                    st.markdown("#### Tabla principal")
                    st.dataframe(filtered[cols_resumidas], use_container_width=True)

                with st.expander("Mostrar detalle de desafíos", expanded=False):
                    st.markdown("### Detalle de desafíos:")
                    for i, row in filtered.iterrows():
                        with st.expander(
                                f"Desafío {row['Inicio']} - Estado: {row['Estado']} - Motivo: {row['Motivo Cierre']}"):
                            st.write(
                                f"Duración: F1={row['Duracion F1']}, F2={row['Duracion F2']}, Fondeado={row['Duracion Funded']}")
                            lista = row["Trades usados detallados"]
                            if len(lista):
                                detalle = pd.DataFrame(lista)
                                detalle = detalle[
                                    ["Close Time", "Profit", "Profit (riesgo fase)", "Capital", "MaxDD", "Cleaned_EA",
                                     "Open Time", "Fase"]]
                                detalle = detalle.round(2)
                                st.dataframe(detalle)
                            else:
                                st.info("No hay trades en este desafío.")

            else:
                st.warning("No hay resultados para los parámetros/configuración actual.")

        else:
            st.info("Introduce los parámetros deseados y pulsa **Procesar desafíos**.")
else:
    st.warning("Sube primero el CSV de trades para comenzar.")

# === LABORATORIO DE RIESGOS ===
st.markdown("---")

with st.expander("Laboratorio de riesgos", expanded=False):
    if uploaded_file is None:
        st.warning("Primero sube un archivo CSV antes de analizar riesgos.")
    else:
        st.subheader("Laboratorio de Riesgos para cada fase")
        st.markdown("Separa por comas los riesgos que quieres probar para cada fase:")

        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            riesgos_fase1_text = st.text_input("Fase 1 (riesgo):", "7.5,10,12.5,15,20", key="riesgo1")
        with colf2:
            riesgos_fase2_text = st.text_input("Fase 2 (riesgo):", "7.5,10,12.5,15", key="riesgo2")
        with colf3:
            riesgos_fondeada_text = st.text_input("Fase Fondeada (riesgo):", "3,5,8,10", key="riesgo3")

        calcular_riesgos = st.button("Calcular valores laboratorio", key="calcula_riesgos")

        params = {
            'pct_fase1': pct_fase1,
            'pct_fase2': pct_fase2,
            'drawdown_diario': drawdown_diario,
            'drawdown_max': drawdown_max,
            'riesgo_fase1': riesgo_fase1,
            'riesgo_fase2': riesgo_fase2,
            'riesgo_fondeado': riesgo_fondeado
        }

        if calcular_riesgos:
            riesgos_f1 = [float(x) for x in riesgos_fase1_text.replace(" ", "").split(",") if x]
            riesgos_f2 = [float(x) for x in riesgos_fase2_text.replace(" ", "").split(",") if x]
            riesgos_fondeada = [float(x) for x in riesgos_fondeada_text.replace(" ", "").split(",") if x]

            st.markdown("### Fase 1 (simulación en batch)")
            res_f1 = []
            for r in riesgos_f1:
                params_aux = params.copy()
                params_aux['riesgo_fase1'] = r
                batch_results = []
                for periodo in meses:
                    mes_ini = pd.Timestamp(periodo.start_time)
                    res = simular_challenge(df, mes_ini, params_aux)
                    if res:
                        batch_results.append(res)
                df_batch = pd.DataFrame(batch_results)
                n_ok = (df_batch["Estado"].isin(["Funded quemada", "Funded (vivo)"])).sum()
                pct_ok = n_ok / len(df_batch) * 100 if len(df_batch) else 0
                res_f1.append({
                    "Riesgo F1": r,
                    "Media duración F1 (trades)": df_batch['Duracion F1'].mean() if len(df_batch) else 0,
                    "Media duración F1 (días)": df_batch['Duracion F1 días'].mean() if 'Duracion F1 días' in df_batch else 0,
                    "% superados": pct_ok
                })
            df_f1 = pd.DataFrame(res_f1).round(2)
            st.markdown('<div class="scrollable-table">' + df_f1.to_html(index=False) + '</div>', unsafe_allow_html=True)

            st.markdown("### Fase 2 (simulación en batch)")
            res_f2 = []
            for r in riesgos_f2:
                params_aux = params.copy()
                params_aux['riesgo_fase2'] = r
                batch_results = []
                for periodo in meses:
                    mes_ini = pd.Timestamp(periodo.start_time)
                    res = simular_challenge(df, mes_ini, params_aux)
                    if res:
                        batch_results.append(res)
                df_batch = pd.DataFrame(batch_results)
                n_ok = (df_batch["Estado"].isin(["Funded quemada", "Funded (vivo)"])).sum()
                pct_ok = n_ok / len(df_batch) * 100 if len(df_batch) else 0
                res_f2.append({
                    "Riesgo F2": r,
                    "Media duración F2 (trades)": df_batch['Duracion F2'].mean() if len(df_batch) else 0,
                    "Media duración F2 (días)": df_batch['Duracion F2 días'].mean() if 'Duracion F2 días' in df_batch else 0,
                    "% superados": pct_ok
                })
            df_f2 = pd.DataFrame(res_f2).round(2)
            st.markdown('<div class="scrollable-table">' + df_f2.to_html(index=False) + '</div>', unsafe_allow_html=True)

            st.markdown("### Fase Fondeada (simulación en batch)")
            res_fon = []
            for r in riesgos_fondeada:
                params_aux = params.copy()
                params_aux['riesgo_fondeado'] = r
                batch_results = []
                for periodo in meses:
                    mes_ini = pd.Timestamp(periodo.start_time)
                    res = simular_challenge(df, mes_ini, params_aux)
                    if res:
                        batch_results.append(res)
                df_batch = pd.DataFrame(batch_results)
                fondeados = df_batch[df_batch["Duracion Funded"] > 0]
                n_fondeadas = fondeados.shape[0]
                n_quemadas = fondeados[fondeados["Estado"] == "Funded quemada"].shape[0]
                pct_quemadas = (n_quemadas / n_fondeadas * 100) if n_fondeadas else 0
                media_profit = fondeados['Profit Final'].mean() if n_fondeadas else 0
                media_duracion_trades = fondeados['Duracion Funded'].mean() if n_fondeadas else 0
                media_duracion_dias = fondeados['Duracion Funded días'].mean() if 'Duracion Funded días' in fondeados and n_fondeadas else 0
                media_profit_mes = (media_profit / (media_duracion_dias / 22)) if media_duracion_dias else 0  # 22 días/mes

                res_fon.append({
                    "Riesgo Funded": r,
                    "Media duración Fondeado (trades)": media_duracion_trades,
                    "Media duración Fondeado (días)": media_duracion_dias,
                    "Media Profit Fondeado": media_profit,
                    "% Fondeados quemados": pct_quemadas,
                    "Media Profit al mes": media_profit_mes
                })
            df_fon = pd.DataFrame(res_fon).round(2)
            st.markdown('<div class="scrollable-table">' + df_fon.to_html(index=False) + '</div>', unsafe_allow_html=True)

        elif calcular_riesgos is not None:

            st.info("Pulsa 'Calcular valores laboratorio' para lanzar la simulación con los riesgos indicados.")
