import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import io
import plotly.express as px
import plotly.graph_objects as go
import os

# Configuración de la página
if os.path.exists('favicon.png'):
    page_icon_config = "favicon.png"
elif os.path.exists('logo_oxxo.png'):
    page_icon_config = "logo_oxxo.png"
else:
    page_icon_config = "🏪"

st.set_page_config(
    page_title="Modelo de Tienda Espejo OXXO",
    page_icon=page_icon_config,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con colores de OXXO
st.markdown("""
    <style>
    :root {
        --oxxo-red: #ED1C24;
        --oxxo-yellow: #FFD100;
        --oxxo-dark: #1a1a1a;
    }
    .main-header {
        background: linear-gradient(135deg, #ED1C24 0%, #C41E3A 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 { color: white; font-size: 2.5rem; font-weight: bold; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-header p { color: #FFD100; font-size: 1.2rem; margin: 0.5rem 0 0 0; }
    .stButton>button { background-color: #ED1C24; color: white; border: none; border-radius: 5px; padding: 0.5rem 2rem; font-weight: bold; transition: all 0.3s; }
    .stButton>button:hover { background-color: #C41E3A; box-shadow: 0 4px 8px rgba(237, 28, 36, 0.3); transform: translateY(-2px); }
    [data-testid="stMetricValue"] { color: #ED1C24; font-weight: bold; }
    [data-testid="stSidebar"] { background: #1a1a1a !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFD100; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: #ffffff !important; }
    [data-testid="stSidebar"] .stCheckbox label { color: #ffffff !important; }
    [data-testid="stSidebar"] .stSlider label { color: #ffffff !important; }
    [data-testid="stSidebar"] .stFileUploader label { color: #ffffff !important; }
    [data-testid="stSidebar"] .streamlit-expanderHeader { background-color: #2a2a2a; color: #FFD100 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f0f0; border-radius: 5px 5px 0 0; padding: 10px 20px; color: #1a1a1a; }
    .stTabs [aria-selected="true"] { background-color: #ED1C24; color: white; }
    .streamlit-expanderHeader { background-color: #f8f9fa; border-radius: 5px; color: #ED1C24; font-weight: 600; }
    .dataframe { border: 2px solid #ED1C24 !important; }
    .stSuccess { background-color: #d4edda; border-left: 4px solid #28a745; }
    hr { border-color: #FFD100; border-width: 2px; }
    [data-testid="stSidebar"] hr { border-color: #ED1C24; border-width: 2px; }
    .metric-card { background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid #ED1C24; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0; }
    .stDownloadButton>button { background-color: #FFD100; color: #1a1a1a; border: none; font-weight: bold; }
    .stDownloadButton>button:hover { background-color: #e6bb00; }
    [data-testid="stSidebar"] .stMarkdown { color: #ffffff; }
    [data-testid="stSidebar"] .stCaption { color: #cccccc !important; }
    </style>
    """, unsafe_allow_html=True)

# Header
col_logo, col_title = st.columns([1, 4])
with col_logo:
    if os.path.exists("logo_oxxo.png"):
        st.image("logo_oxxo.png", width=150)
with col_title:
    st.markdown("""
        <div class='main-header'>
            <h1> 🏪 Modelo de Tienda Espejo</h1>
            <p>Encuentra la tienda operativa más similar a tu propuesta</p>
        </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODELO ESTADÍSTICO
# Columnas reales del Excel: VU6M (ventas últimos 6 meses), TRU6 (tráfico últimos 6 meses)
# ──────────────────────────────────────────────
def calcular_tienda_espejo_estadistico(df, nueva_tienda, pesos=None):
    """
    Distancia euclidiana ponderada normalizada.
    Variables numéricas: ESTRATO, AREA, VIVIENDAS, EMPLEOS, VU6M, TRU6
    Variables categóricas: ZONA, TIPO DE LOCAL, GENERADOR, MUN
    """
    if pesos is None:
        pesos = {
            'SEG26': 0.30,
            'ZONA': 0.10,
            'ESTRATO': 0.08,
            'TIPO DE LOCAL': 0.07,
            'AREA': 0.08,
            'GENERADOR': 0.07,
            'MUN': 0.06,
            'VIVIENDAS': 0.06,
            'EMPLEOS': 0.06,
            'VU6M': 0.12,
            'TRU6': 0.10
        }

    df_filtrado = df[df['SEG26'] == nueva_tienda['SEG26']].copy()

    if len(df_filtrado) == 0:
        return None, "No se encontraron tiendas en el mismo segmento"

    # ── Variables numéricas ──
    vars_numericas = ['ESTRATO', 'AREA', 'VIVIENDAS', 'EMPLEOS', 'VU6M', 'TRU6']

    for v in vars_numericas:
        if v not in df_filtrado.columns:
            df_filtrado[v] = 0

    X_num_df = df_filtrado[vars_numericas].fillna(0)
    X_num_nueva = np.array([[
        nueva_tienda['ESTRATO'],
        nueva_tienda['AREA'],
        nueva_tienda['VIVIENDAS'],
        nueva_tienda['EMPLEOS'],
        nueva_tienda['VU6M'],
        nueva_tienda['TRU6']
    ]])

    scaler = StandardScaler()
    X_num_df_scaled = scaler.fit_transform(X_num_df)
    X_num_nueva_scaled = scaler.transform(X_num_nueva)

    # ── Variables categóricas ──
    vars_categoricas = ['ZONA', 'TIPO DE LOCAL', 'GENERADOR', 'MUN']
    X_cat_df = np.zeros((len(df_filtrado), len(vars_categoricas)))
    for i, var in enumerate(vars_categoricas):
        X_cat_df[:, i] = (df_filtrado[var] == nueva_tienda[var]).astype(int)
    X_cat_nueva = np.ones((1, len(vars_categoricas)))

    # ── Combinar ──
    X_df_completo = np.hstack([X_num_df_scaled, X_cat_df])
    X_nueva_completo = np.hstack([X_num_nueva_scaled, X_cat_nueva])

    # ── Pesos (orden: ESTRATO, AREA, VIVIENDAS, EMPLEOS, VU6M, TRU6, ZONA, TIPO, GEN, MUN) ──
    peso_vector = np.array([
        pesos.get('ESTRATO', 0.08),
        pesos.get('AREA', 0.08),
        pesos.get('VIVIENDAS', 0.06),
        pesos.get('EMPLEOS', 0.06),
        pesos.get('VU6M', 0.12),
        pesos.get('TRU6', 0.10),
        pesos.get('ZONA', 0.10),
        pesos.get('TIPO DE LOCAL', 0.07),
        pesos.get('GENERADOR', 0.07),
        pesos.get('MUN', 0.06),
    ])

    X_df_ponderado = X_df_completo * np.sqrt(peso_vector)
    X_nueva_ponderado = X_nueva_completo * np.sqrt(peso_vector)

    distancias = euclidean_distances(X_nueva_ponderado, X_df_ponderado)[0]

    max_dist = np.max(distancias)
    min_dist = np.min(distancias)
    if max_dist > min_dist:
        distancias_norm = (distancias - min_dist) / (max_dist - min_dist)
        similitud_scores = (1 - distancias_norm) * 100
    else:
        similitud_scores = np.full_like(distancias, 100.0)

    df_resultado = df_filtrado.copy()
    df_resultado['DISTANCIA'] = distancias
    df_resultado['SIMILITUD'] = similitud_scores
    df_resultado = df_resultado.sort_values('SIMILITUD', ascending=False)

    return df_resultado, None


def calcular_estadisticas(df_resultado, nueva_tienda):
    top_10 = df_resultado.head(10)

    renta_col = None
    for col in df_resultado.columns:
        if 'RENTA' in col.upper():
            renta_col = col
            break

    stats = {
        'VT_promedio': top_10['VT'].mean(),
        'VT_std': top_10['VT'].std(),
        'ET_promedio': top_10['ET'].mean(),
        'ET_std': top_10['ET'].std(),
        'VU6M_promedio': top_10['VU6M'].mean() if 'VU6M' in top_10.columns else 0,
        'VU6M_std': top_10['VU6M'].std() if 'VU6M' in top_10.columns else 0,
        'TRU6_promedio': top_10['TRU6'].mean() if 'TRU6' in top_10.columns else 0,
        'TRU6_std': top_10['TRU6'].std() if 'TRU6' in top_10.columns else 0,
        'RENTA_promedio': top_10[renta_col].mean() if renta_col and renta_col in top_10.columns else 0,
        'RENTA_std': top_10[renta_col].std() if renta_col and renta_col in top_10.columns else 0,
        'AREA_promedio': top_10['AREA'].mean(),
        'similitud_promedio': top_10['SIMILITUD'].mean(),
        'renta_col': renta_col if renta_col else 'RENTA',
    }
    return stats


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Cargar Datos")

    usar_ejemplo = st.checkbox("Usar datos precargados", value=True)

    if usar_ejemplo:
        try:
            df = pd.read_excel('Book.xlsx')
            st.markdown(f"""
                <div style='background-color: #ED1C24; padding: 0.8rem; border-radius: 5px; 
                            color: white; border-left: 4px solid #FFD100;'>
                    ✅ <strong>{len(df)}</strong> tiendas cargadas
                </div>
            """, unsafe_allow_html=True)
        except:
            st.warning("⚠️ Carga tu archivo Excel abajo")
            usar_ejemplo = False

    if not usar_ejemplo:
        uploaded_file = st.file_uploader("Sube tu archivo Excel", type=['xlsx', 'xls'])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.markdown(f"""
                <div style='background-color: #ED1C24; padding: 0.8rem; border-radius: 5px; 
                            color: white; border-left: 4px solid #FFD100;'>
                    ✅ <strong>{len(df)}</strong> tiendas cargadas
                </div>
            """, unsafe_allow_html=True)
        else:
            df = None

    if df is not None:
        st.divider()
        st.header("⚙️ Configuración de Pesos")
        st.caption("Ajusta la importancia de cada característica")
        st.markdown("""
            <div style='background-color: #FFD100; padding: 0.8rem; border-radius: 5px; 
                        color: #1a1a1a; border-left: 4px solid #ED1C24;'>
                💡 Los pesos se normalizan automáticamente para sumar 100%
            </div>
        """, unsafe_allow_html=True)

        peso_seg       = st.slider("Segmento (filtro obligatorio)", 0, 100, 30, disabled=True)
        peso_zona      = st.slider("Zona geográfica", 0, 100, 10)
        peso_estrato   = st.slider("Estrato socioeconómico", 0, 100, 8)
        peso_tipo      = st.slider("Tipo de Local", 0, 100, 7)
        peso_area      = st.slider("Área (m²)", 0, 100, 8)
        peso_generador = st.slider("Generador", 0, 100, 7)
        peso_mun       = st.slider("Municipio", 0, 100, 6)
        peso_viviendas = st.slider("Viviendas Totales (VT)", 0, 100, 6)
        peso_empleos   = st.slider("Empleos Totales (ET)", 0, 100, 6)

        st.markdown("---")
        st.markdown("**📊 Variables de Rendimiento (últimos 6 meses)**")
        peso_vu6m = st.slider("💰 Ventas Últ. 6 Meses (VU6M)", 0, 100, 12)
        peso_tru6 = st.slider("🚶 Tráfico Últ. 6 Meses (TRU6)", 0, 100, 10)

        total = (peso_zona + peso_estrato + peso_tipo + peso_area + peso_generador +
                 peso_mun + peso_viviendas + peso_empleos + peso_vu6m + peso_tru6)

        if total > 0:
            pesos = {
                'SEG26': 0.30,
                'ZONA':          peso_zona      / total * 0.70,
                'ESTRATO':       peso_estrato   / total * 0.70,
                'TIPO DE LOCAL': peso_tipo      / total * 0.70,
                'AREA':          peso_area      / total * 0.70,
                'GENERADOR':     peso_generador / total * 0.70,
                'MUN':           peso_mun       / total * 0.70,
                'VIVIENDAS':     peso_viviendas / total * 0.70,
                'EMPLEOS':       peso_empleos   / total * 0.70,
                'VU6M':          peso_vu6m      / total * 0.70,
                'TRU6':          peso_tru6      / total * 0.70,
            }
        else:
            pesos = None

        with st.expander("Ver pesos normalizados"):
            if pesos:
                for key, val in pesos.items():
                    st.write(f"**{key}:** {val*100:.1f}%")


# ──────────────────────────────────────────────
# CONTENIDO PRINCIPAL
# ──────────────────────────────────────────────
if df is not None:

    df['VIVIENDAS'] = df['VT']
    df['EMPLEOS']   = df['ET']

    # Columnas VU6M y TRU6: si no existen en el Excel, iniciar en 0
    if 'VU6M' not in df.columns:
        df['VU6M'] = 0
        st.warning("⚠️ No se encontró la columna **VU6M** (Ventas últimos 6 meses) en el Excel. Se usará 0.")
    if 'TRU6' not in df.columns:
        df['TRU6'] = 0
        st.warning("⚠️ No se encontró la columna **TRU6** (Tráfico últimos 6 meses) en el Excel. Se usará 0.")

    renta_col_disponible = None
    for col in df.columns:
        if 'RENTA' in col.upper():
            renta_col_disponible = col
            break
    if renta_col_disponible is None:
        df['RENTA'] = 0
        renta_col_disponible = 'RENTA'

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📝 Nueva Tienda Propuesta")

        with st.form("form_nueva_tienda"):
            nombre_nueva = st.text_input("Nombre de la tienda propuesta", "Mi Nueva Tienda")

            st.markdown("##### Características Principales")
            segmento   = st.selectbox("Segmento (SEG26)",   options=sorted(df['SEG26'].unique()))
            zona       = st.selectbox("Zona",               options=sorted(df['ZONA'].unique()))
            municipio  = st.selectbox("Municipio",          options=sorted(df['MUN'].unique()))
            estrato    = st.selectbox("Estrato",            options=sorted(df['ESTRATO'].unique()))
            tipo_local = st.selectbox("Tipo de Local",      options=sorted(df['TIPO DE LOCAL'].unique()))
            generador  = st.selectbox("Generador",          options=sorted(df['GENERADOR'].unique()))

            st.markdown("##### Métricas Numéricas")
            col_a, col_b = st.columns(2)
            with col_a:
                area      = st.number_input("Área (m²)",         min_value=0.0, value=100.0, step=10.0)
                viviendas = st.number_input("Viviendas Totales", min_value=0,   value=1000,  step=100)
                empleos   = st.number_input("Empleos Totales",   min_value=0,   value=500,   step=50)
            with col_b:
                vu6m = st.number_input(
                    "💰 Ventas Últ. 6 Meses ($)",
                    min_value=0.0, value=0.0, step=1000.0,
                    help="Venta acumulada de los últimos 6 meses (VU6M) esperada para la nueva tienda"
                )
                tru6 = st.number_input(
                    "🚶 Tráfico Últ. 6 Meses",
                    min_value=0, value=0, step=100,
                    help="Número de personas estimadas en los últimos 6 meses (TRU6)"
                )

            submitted = st.form_submit_button("🔍 Buscar Tienda Espejo", use_container_width=True)

    with col2:
        st.subheader("🎯 Resultados")

        if submitted:
            nueva_tienda = {
                'NAME':      nombre_nueva,
                'SEG26':     segmento,
                'ZONA':      zona,
                'MUN':       municipio,
                'ESTRATO':   estrato,
                'TIPO DE LOCAL': tipo_local,
                'AREA':      area,
                'GENERADOR': generador,
                'VIVIENDAS': viviendas,
                'EMPLEOS':   empleos,
                'VU6M':      vu6m,
                'TRU6':      tru6,
            }

            resultado, error = calcular_tienda_espejo_estadistico(df, nueva_tienda, pesos)

            if error:
                st.error(error)
            else:
                stats     = calcular_estadisticas(resultado, nueva_tienda)
                renta_col = stats['renta_col']

                st.success("✅ Tiendas espejo encontradas usando modelo estadístico")

                mejor = resultado.iloc[0]
                st.markdown("### 🏆 Mejor Tienda Espejo")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Nombre", mejor['NAME'])
                    st.metric("Código", mejor['CR'])
                with c2:
                    st.metric("Similitud",  f"{mejor['SIMILITUD']:.1f}%")
                    st.metric("Distancia",  f"{mejor['DISTANCIA']:.3f}")
                with c3:
                    st.metric("Viviendas (VT)", f"{mejor['VT']:,.0f}")
                    st.metric("Empleos (ET)",   f"{mejor['ET']:,.0f}")
                with c4:
                    vu6m_val = mejor['VU6M'] if 'VU6M' in mejor.index else 0
                    tru6_val = mejor['TRU6'] if 'TRU6' in mejor.index else 0
                    st.metric("💰 Ventas U6M",  f"${vu6m_val:,.0f}")
                    st.metric("🚶 Tráfico U6M", f"{tru6_val:,.0f}")

                with st.expander("📊 Ver detalles completos de la mejor tienda", expanded=False):
                    col_det1, col_det2 = st.columns(2)
                    with col_det1:
                        st.write(f"**Segmento:** {mejor['SEG26']}")
                        st.write(f"**Zona:** {mejor['ZONA']}")
                        st.write(f"**Municipio:** {mejor['MUN']}")
                        st.write(f"**Estrato:** {mejor['ESTRATO']}")
                        st.write(f"**💰 Ventas Últ. 6 Meses (VU6M):** ${vu6m_val:,.0f}")
                    with col_det2:
                        st.write(f"**Tipo de Local:** {mejor['TIPO DE LOCAL']}")
                        st.write(f"**Generador:** {mejor['GENERADOR']}")
                        st.write(f"**Viviendas (VT):** {mejor['VT']:,.0f}")
                        st.write(f"**Empleos (ET):** {mejor['ET']:,.0f}")
                        st.write(f"**🚶 Tráfico Últ. 6 Meses (TRU6):** {tru6_val:,.0f}")

                st.divider()

                # Estadísticas Top 10
                st.markdown("### 📈 Estadísticas del Top 10")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Viviendas Prom (VT)", f"{stats['VT_promedio']:,.0f}")
                    st.caption(f"±{stats['VT_std']:,.0f}")
                with col_s2:
                    st.metric("Empleos Prom (ET)", f"{stats['ET_promedio']:,.0f}")
                    st.caption(f"±{stats['ET_std']:,.0f}")
                with col_s3:
                    st.metric("💰 Ventas U6M Prom", f"${stats['VU6M_promedio']:,.0f}")
                    st.caption(f"±{stats['VU6M_std']:,.0f}")
                with col_s4:
                    st.metric("🚶 Tráfico U6M Prom", f"{stats['TRU6_promedio']:,.0f}")
                    st.caption(f"Similitud: {stats['similitud_promedio']:.1f}%")

                st.divider()

                # Top 10
                st.markdown("### 📋 Top 10 Alternativas")

                columnas_mostrar = ['CR', 'NAME', 'ZONA', 'MUN', 'ESTRATO',
                                    'TIPO DE LOCAL', 'AREA', 'VT', 'ET',
                                    'VU6M', 'TRU6', 'SIMILITUD', 'DISTANCIA']

                if renta_col in resultado.columns and renta_col not in columnas_mostrar:
                    columnas_mostrar.insert(-2, renta_col)

                columnas_mostrar = [c for c in columnas_mostrar if c in resultado.columns]

                top_10     = resultado.head(10)[columnas_mostrar]
                top_10_display = top_10.copy()

                top_10_display['SIMILITUD'] = top_10_display['SIMILITUD'].apply(lambda x: f"{x:.1f}%")
                top_10_display['DISTANCIA'] = top_10_display['DISTANCIA'].apply(lambda x: f"{x:.3f}")
                top_10_display['AREA']      = top_10_display['AREA'].apply(lambda x: f"{x:.1f}")
                top_10_display['VT']        = top_10_display['VT'].apply(lambda x: f"{x:,.0f}")
                top_10_display['ET']        = top_10_display['ET'].apply(lambda x: f"{x:,.0f}")
                if 'VU6M' in top_10_display.columns:
                    top_10_display['VU6M']  = top_10_display['VU6M'].apply(lambda x: f"${x:,.0f}")
                if 'TRU6' in top_10_display.columns:
                    top_10_display['TRU6']  = top_10_display['TRU6'].apply(lambda x: f"{x:,.0f}")
                if renta_col in top_10_display.columns:
                    top_10_display[renta_col] = top_10_display[renta_col].apply(lambda x: f"${x:,.0f}")

                top_10_display = top_10_display.rename(columns={
                    'VT':   'Viviendas (VT)',
                    'ET':   'Empleos (ET)',
                    'VU6M': '💰 Ventas U6M ($)',
                    'TRU6': '🚶 Tráfico U6M'
                })

                st.dataframe(top_10_display, use_container_width=True, hide_index=True)

                csv = resultado.head(20).to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Top 20 (CSV)",
                    data=csv,
                    file_name=f"tiendas_espejo_{nombre_nueva.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

                st.divider()

                # Visualizaciones
                st.markdown("### 📊 Análisis Visual")

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Comparación de Métricas",
                    "Ventas & Tráfico U6M",
                    "Distribución Geográfica",
                    "Análisis de Similitud",
                    "Modelo Estadístico"
                ])

                with tab1:
                    top_5 = resultado.head(5)
                    fig_metricas = go.Figure()
                    fig_metricas.add_trace(go.Bar(name='Viviendas (VT)', x=top_5['NAME'], y=top_5['VT'], marker_color='#ED1C24'))
                    fig_metricas.add_trace(go.Bar(name='Empleos (ET)',   x=top_5['NAME'], y=top_5['ET'], marker_color='#FFD100'))
                    fig_metricas.update_layout(title='Top 5 - Viviendas vs Empleos', barmode='group', height=400,
                                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_metricas, use_container_width=True)

                with tab2:
                    top_10_raw = resultado.head(10)

                    # Ventas U6M por tienda
                    if 'VU6M' in top_10_raw.columns:
                        fig_venta = go.Figure()
                        fig_venta.add_trace(go.Bar(
                            name='Ventas Últ. 6 Meses ($)',
                            x=top_10_raw['NAME'],
                            y=top_10_raw['VU6M'],
                            marker_color='#ED1C24'
                        ))
                        if vu6m > 0:
                            fig_venta.add_hline(
                                y=vu6m,
                                line_dash="dash",
                                line_color="#FFD100",
                                annotation_text=f"Tu propuesta: ${vu6m:,.0f}",
                                annotation_position="top left"
                            )
                        fig_venta.update_layout(
                            title='💰 Ventas Últimos 6 Meses – Tiendas Espejo (Top 10)',
                            xaxis_tickangle=-45, height=400,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_venta, use_container_width=True)

                    # Tráfico U6M por tienda
                    if 'TRU6' in top_10_raw.columns:
                        fig_traf = go.Figure()
                        fig_traf.add_trace(go.Bar(
                            name='Tráfico Últ. 6 Meses',
                            x=top_10_raw['NAME'],
                            y=top_10_raw['TRU6'],
                            marker_color='#FFD100'
                        ))
                        if tru6 > 0:
                            fig_traf.add_hline(
                                y=tru6,
                                line_dash="dash",
                                line_color="#ED1C24",
                                annotation_text=f"Tu propuesta: {tru6:,}",
                                annotation_position="top left"
                            )
                        fig_traf.update_layout(
                            title='🚶 Tráfico Últimos 6 Meses – Tiendas Espejo (Top 10)',
                            xaxis_tickangle=-45, height=400,
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_traf, use_container_width=True)

                    # Scatter Ventas vs Tráfico
                    if 'VU6M' in top_10_raw.columns and 'TRU6' in top_10_raw.columns:
                        fig_vt = px.scatter(
                            top_10_raw,
                            x='TRU6',
                            y='VU6M',
                            size='AREA',
                            color='SIMILITUD',
                            hover_data=['NAME', 'ZONA'],
                            title='Ventas vs Tráfico U6M (Tamaño = Área, Color = Similitud)',
                            labels={'TRU6': 'Tráfico Últ. 6 Meses', 'VU6M': 'Ventas U6M ($)'},
                            color_continuous_scale=['#C41E3A', '#ED1C24', '#FFD100', '#28a745']
                        )
                        if vu6m > 0 or tru6 > 0:
                            fig_vt.add_trace(go.Scatter(
                                x=[tru6], y=[vu6m],
                                mode='markers',
                                marker=dict(color='blue', size=14, symbol='star'),
                                name='Tu propuesta'
                            ))
                        fig_vt.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_vt, use_container_width=True)

                with tab3:
                    colors_oxxo = ['#ED1C24', '#FFD100', '#C41E3A', '#FFA500', '#FF6B6B']
                    dist_zona = top_10_raw['ZONA'].value_counts()
                    fig_zona = px.pie(values=dist_zona.values, names=dist_zona.index,
                                     title='Distribución por Zona', color_discrete_sequence=colors_oxxo)
                    fig_zona.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_zona, use_container_width=True)

                    dist_estrato = top_10_raw['ESTRATO'].value_counts().sort_index()
                    fig_estrato = px.bar(x=dist_estrato.index, y=dist_estrato.values,
                                        title='Distribución por Estrato',
                                        labels={'x': 'Estrato', 'y': 'Cantidad'},
                                        color=dist_estrato.values,
                                        color_continuous_scale=['#FFD100', '#FFA500', '#ED1C24', '#C41E3A'])
                    fig_estrato.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_estrato, use_container_width=True)

                with tab4:
                    fig_sim = px.bar(top_10_raw, x='NAME', y='SIMILITUD',
                                     title='% Similitud - Top 10',
                                     labels={'NAME': 'Tienda', 'SIMILITUD': 'Similitud (%)'},
                                     color='SIMILITUD',
                                     color_continuous_scale=['#C41E3A', '#ED1C24', '#FFD100', '#28a745'])
                    fig_sim.update_layout(xaxis_tickangle=-45, height=400,
                                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_sim, use_container_width=True)

                    st.markdown("#### 📋 Comparación con Tienda Espejo")
                    vu6m_val2 = mejor.get('VU6M', 0)
                    tru6_val2 = mejor.get('TRU6', 0)
                    comparacion = pd.DataFrame({
                        'Característica': ['Segmento', 'Zona', 'Municipio', 'Estrato',
                                           'Tipo de Local', 'Generador', 'Área',
                                           'Viviendas (VT)', 'Empleos (ET)',
                                           '💰 Ventas U6M ($)', '🚶 Tráfico U6M'],
                        'Tu Propuesta': [
                            nueva_tienda['SEG26'], nueva_tienda['ZONA'], nueva_tienda['MUN'],
                            nueva_tienda['ESTRATO'], nueva_tienda['TIPO DE LOCAL'], nueva_tienda['GENERADOR'],
                            f"{nueva_tienda['AREA']:.1f} m²",
                            f"{nueva_tienda['VIVIENDAS']:,}", f"{nueva_tienda['EMPLEOS']:,}",
                            f"${nueva_tienda['VU6M']:,.0f}",
                            f"{nueva_tienda['TRU6']:,}"
                        ],
                        'Tienda Espejo': [
                            mejor['SEG26'], mejor['ZONA'], mejor['MUN'],
                            mejor['ESTRATO'], mejor['TIPO DE LOCAL'], mejor['GENERADOR'],
                            f"{mejor['AREA']:.1f} m²",
                            f"{mejor['VT']:,.0f}", f"{mejor['ET']:,.0f}",
                            f"${vu6m_val2:,.0f}",
                            f"{tru6_val2:,.0f}"
                        ],
                        'Coincide / Diferencia': [
                            '✅' if nueva_tienda['SEG26'] == mejor['SEG26'] else '❌',
                            '✅' if nueva_tienda['ZONA'] == mejor['ZONA'] else '❌',
                            '✅' if nueva_tienda['MUN'] == mejor['MUN'] else '❌',
                            '✅' if nueva_tienda['ESTRATO'] == mejor['ESTRATO'] else '❌',
                            '✅' if nueva_tienda['TIPO DE LOCAL'] == mejor['TIPO DE LOCAL'] else '❌',
                            '✅' if nueva_tienda['GENERADOR'] == mejor['GENERADOR'] else '❌',
                            f"{abs(nueva_tienda['AREA'] - mejor['AREA']):.1f} m²",
                            f"{abs(nueva_tienda['VIVIENDAS'] - mejor['VT']):,.0f}",
                            f"{abs(nueva_tienda['EMPLEOS'] - mejor['ET']):,.0f}",
                            f"${abs(nueva_tienda['VU6M'] - vu6m_val2):,.0f}",
                            f"{abs(nueva_tienda['TRU6'] - tru6_val2):,.0f}"
                        ]
                    })
                    st.dataframe(comparacion, use_container_width=True, hide_index=True)

                with tab5:
                    st.markdown("#### 🔬 Modelo Estadístico: Distancia Euclidiana Ponderada")
                    st.markdown("""
                    **Metodología:**
                    1. **Filtrado** por segmento (SEG26)
                    2. **Normalización** de variables numéricas (StandardScaler μ=0, σ=1)
                    3. **Codificación** binaria de variables categóricas
                    4. **Ponderación** configurable por el usuario
                    5. **Distancia euclidiana** en espacio multidimensional
                    6. **Similitud** = inversión normalizada a 0-100%
                    
                    **Variables numéricas:** ESTRATO, ÁREA, VIVIENDAS, EMPLEOS, **VU6M** (Ventas Últ. 6 Meses), **TRU6** (Tráfico Últ. 6 Meses)
                    
                    **Variables categóricas:** ZONA, TIPO DE LOCAL, GENERADOR, MUNICIPIO
                    """)

                    fig_dist = px.histogram(resultado.head(50), x='DISTANCIA', nbins=20,
                                            title='Distribución de Distancias (Top 50)',
                                            color_discrete_sequence=['#ED1C24'])
                    fig_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_dist, use_container_width=True)

                    fig_rel = px.scatter(resultado.head(30), x='DISTANCIA', y='SIMILITUD',
                                         hover_data=['NAME'],
                                         title='Distancia vs Similitud (Top 30)',
                                         color_discrete_sequence=['#ED1C24'])
                    fig_rel.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_rel, use_container_width=True)

else:
    st.info("👈 Por favor, carga un archivo Excel en la barra lateral para comenzar")
    st.markdown("""
    ### 📖 Cómo usar esta herramienta:
    1. **Carga tu archivo Excel** en la barra lateral (o usa los datos precargados)
    2. **Completa los datos** de la nueva tienda, incluyendo **VU6M** y **TRU6**
    3. **Ajusta los pesos** en el sidebar
    4. **Haz clic en "Buscar Tienda Espejo"**
    
    ### 📊 Columnas requeridas en el Excel:
    - **VU6M** — Ventas acumuladas de los últimos 6 meses por tienda
    - **TRU6** — Tráfico acumulado de los últimos 6 meses por tienda
    
    Asegúrate de que tu archivo Excel incluya estas columnas con esos nombres exactos.
    """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #ED1C24 0%, #C41E3A 100%); border-radius: 10px;'>
        <h3 style='color: #FFD100; margin: 0;'>🏪 Modelo de Tienda Espejo OXXO</h3>
        <p style='color: white; margin: 0.5rem 0 0 0;'>v4.0 | VU6M & TRU6 (Últimos 6 Meses)</p>
    </div>
""", unsafe_allow_html=True)
