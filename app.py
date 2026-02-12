import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import io
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Modelo de Tienda Espejo",
    page_icon="🏪",
    layout="wide"
)

# Título
st.title("🏪 Modelo de Tienda Espejo")
st.markdown("### Encuentra la tienda operativa más similar a tu propuesta")

# Función para calcular similitud
def calcular_tienda_espejo(df, nueva_tienda, pesos=None):
    """
    Calcula las tiendas más similares basándose en múltiples características
    """
    if pesos is None:
        pesos = {
            'SEG26': 0.30,
            'ZONA': 0.15,
            'ESTRATO': 0.15,
            'TIPO DE LOCAL': 0.10,
            'AREA': 0.10,
            'GENERADOR': 0.10,
            'MUN': 0.10
        }
    
    # Filtrar tiendas del mismo segmento (obligatorio)
    df_filtrado = df[df['SEG26'] == nueva_tienda['SEG26']].copy()
    
    if len(df_filtrado) == 0:
        return None, "No se encontraron tiendas en el mismo segmento"
    
    # Preparar características para comparación
    caracteristicas = []
    
    # Variables categóricas
    cat_vars = ['ZONA', 'TIPO DE LOCAL', 'GENERADOR', 'MUN']
    for var in cat_vars:
        # Crear encoding simple: 1 si coincide, 0 si no
        caracteristicas.append((df_filtrado[var] == nueva_tienda[var]).astype(int).values)
    
    # Variables numéricas (normalizadas)
    # ESTRATO
    estrato_diff = np.abs(df_filtrado['ESTRATO'] - nueva_tienda['ESTRATO'])
    estrato_sim = 1 / (1 + estrato_diff)  # Similitud inversa a la diferencia
    caracteristicas.append(estrato_sim.values)
    
    # AREA (usando similitud relativa)
    area_diff = np.abs(df_filtrado['AREA'] - nueva_tienda['AREA']) / nueva_tienda['AREA']
    area_sim = 1 / (1 + area_diff)
    caracteristicas.append(area_sim.values)
    
    # Combinar todas las características en una matriz
    X = np.column_stack(caracteristicas)
    
    # Calcular score de similitud ponderado
    peso_values = [
        pesos['ZONA'],
        pesos['TIPO DE LOCAL'],
        pesos['GENERADOR'],
        pesos['MUN'],
        pesos['ESTRATO'],
        pesos['AREA']
    ]
    
    scores = np.average(X, axis=1, weights=peso_values)
    
    # Añadir scores al dataframe
    df_resultado = df_filtrado.copy()
    df_resultado['SIMILITUD'] = scores * 100
    
    # Ordenar por similitud
    df_resultado = df_resultado.sort_values('SIMILITUD', ascending=False)
    
    return df_resultado, None

# Sidebar para cargar datos
with st.sidebar:
    st.header("📂 Cargar Datos")
    
    # Opción para usar archivo de ejemplo o cargar uno nuevo
    usar_ejemplo = st.checkbox("Usar datos precargados", value=True)
    
    if usar_ejemplo:
        try:
            df = pd.read_excel('/mnt/user-data/uploads/Book.xlsx')
            st.success(f"✅ {len(df)} tiendas cargadas")
        except:
            st.warning("⚠️ Carga tu archivo Excel abajo")
            usar_ejemplo = False
    
    if not usar_ejemplo:
        uploaded_file = st.file_uploader("Sube tu archivo Excel", type=['xlsx', 'xls'])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ {len(df)} tiendas cargadas")
        else:
            df = None
    
    if df is not None:
        st.divider()
        st.header("⚙️ Configuración de Pesos")
        st.caption("Ajusta la importancia de cada característica")
        
        peso_seg = st.slider("Segmento (obligatorio)", 0, 100, 30, disabled=True)
        peso_zona = st.slider("Zona", 0, 100, 15)
        peso_estrato = st.slider("Estrato", 0, 100, 15)
        peso_tipo = st.slider("Tipo de Local", 0, 100, 10)
        peso_area = st.slider("Área", 0, 100, 10)
        peso_generador = st.slider("Generador", 0, 100, 10)
        peso_mun = st.slider("Municipio", 0, 100, 10)
        
        # Normalizar pesos
        total = peso_zona + peso_estrato + peso_tipo + peso_area + peso_generador + peso_mun
        pesos = {
            'SEG26': 0.30,  # Fijo
            'ZONA': peso_zona / total * 0.70,
            'ESTRATO': peso_estrato / total * 0.70,
            'TIPO DE LOCAL': peso_tipo / total * 0.70,
            'AREA': peso_area / total * 0.70,
            'GENERADOR': peso_generador / total * 0.70,
            'MUN': peso_mun / total * 0.70
        }

# Contenido principal
if df is not None:
    # Dos columnas: entrada de datos y resultados
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Nueva Tienda Propuesta")
        
        # Formulario para ingresar datos de la nueva tienda
        with st.form("form_nueva_tienda"):
            nombre_nueva = st.text_input("Nombre de la tienda propuesta", "Mi Nueva Tienda")
            
            segmento = st.selectbox("Segmento (SEG26)", options=sorted(df['SEG26'].unique()))
            zona = st.selectbox("Zona", options=sorted(df['ZONA'].unique()))
            municipio = st.selectbox("Municipio", options=sorted(df['MUN'].unique()))
            estrato = st.selectbox("Estrato", options=sorted(df['ESTRATO'].unique()))
            tipo_local = st.selectbox("Tipo de Local", options=sorted(df['TIPO DE LOCAL'].unique()))
            area = st.number_input("Área (m²)", min_value=0.0, value=100.0, step=10.0)
            generador = st.selectbox("Generador", options=sorted(df['GENERADOR'].unique()))
            
            submitted = st.form_submit_button("🔍 Buscar Tienda Espejo", use_container_width=True)
    
    with col2:
        st.subheader("🎯 Resultados")
        
        if submitted:
            # Crear diccionario con la nueva tienda
            nueva_tienda = {
                'NAME': nombre_nueva,
                'SEG26': segmento,
                'ZONA': zona,
                'MUN': municipio,
                'ESTRATO': estrato,
                'TIPO DE LOCAL': tipo_local,
                'AREA': area,
                'GENERADOR': generador
            }
            
            # Calcular tiendas espejo
            resultado, error = calcular_tienda_espejo(df, nueva_tienda, pesos)
            
            if error:
                st.error(error)
            else:
                # Mostrar top 5
                st.success("✅ Tiendas espejo encontradas")
                
                # Tarjeta de la mejor tienda espejo
                mejor = resultado.iloc[0]
                
                st.markdown("### 🏆 Mejor Tienda Espejo")
                
                # Crear tres columnas para info destacada
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Nombre", mejor['NAME'])
                    st.metric("Código", mejor['CR'])
                with c2:
                    st.metric("Similitud", f"{mejor['SIMILITUD']:.1f}%")
                    st.metric("VT", f"{mejor['VT']:,}")
                with c3:
                    st.metric("ET", f"{mejor['ET']:,}")
                    st.metric("Renta", f"${mejor[' RENTA ']:,}")
                
                # Detalles de la tienda espejo
                st.markdown("#### 📊 Características de la Tienda Espejo")
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write(f"**Zona:** {mejor['ZONA']}")
                    st.write(f"**Municipio:** {mejor['MUN']}")
                    st.write(f"**Estrato:** {mejor['ESTRATO']}")
                    st.write(f"**Segmento:** {mejor['SEG26']}")
                
                with info_col2:
                    st.write(f"**Tipo de Local:** {mejor['TIPO DE LOCAL']}")
                    st.write(f"**Área:** {mejor['AREA']:.1f} m²")
                    st.write(f"**Generador:** {mejor['GENERADOR']}")
                
                st.divider()
                
                # Top 10 alternativas
                st.markdown("### 📋 Top 10 Alternativas")
                
                # Preparar dataframe para mostrar
                top_10 = resultado.head(10)[['CR', 'NAME', 'ZONA', 'MUN', 'ESTRATO', 
                                              'TIPO DE LOCAL', 'AREA', 'VT', 'ET', 
                                              ' RENTA ', 'SIMILITUD']]
                
                # Formatear columnas
                top_10_display = top_10.copy()
                top_10_display['SIMILITUD'] = top_10_display['SIMILITUD'].apply(lambda x: f"{x:.1f}%")
                top_10_display['AREA'] = top_10_display['AREA'].apply(lambda x: f"{x:.1f}")
                top_10_display['VT'] = top_10_display['VT'].apply(lambda x: f"{x:,}")
                top_10_display['ET'] = top_10_display['ET'].apply(lambda x: f"{x:,}")
                top_10_display[' RENTA '] = top_10_display[' RENTA '].apply(lambda x: f"${x:,}")
                
                st.dataframe(top_10_display, use_container_width=True, hide_index=True)
                
                # Botón de descarga
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
                
                # Tabs para diferentes visualizaciones
                tab1, tab2, tab3 = st.tabs(["Comparación de Métricas", "Distribución Geográfica", "Análisis de Similitud"])
                
                with tab1:
                    # Comparación de métricas principales
                    top_5 = resultado.head(5)
                    
                    fig_metricas = go.Figure()
                    
                    # VT
                    fig_metricas.add_trace(go.Bar(
                        name='VT (Ventas)',
                        x=top_5['NAME'],
                        y=top_5['VT'],
                        marker_color='lightblue'
                    ))
                    
                    # ET
                    fig_metricas.add_trace(go.Bar(
                        name='ET (Tráfico)',
                        x=top_5['NAME'],
                        y=top_5['ET'],
                        marker_color='lightgreen'
                    ))
                    
                    fig_metricas.update_layout(
                        title='Top 5 Tiendas Espejo - Comparación VT vs ET',
                        xaxis_title='Tienda',
                        yaxis_title='Valor',
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig_metricas, use_container_width=True)
                    
                    # Gráfico de renta vs área
                    fig_renta = px.scatter(
                        top_10,
                        x='AREA',
                        y=' RENTA ',
                        size='VT',
                        color='SIMILITUD',
                        hover_data=['NAME', 'ZONA', 'ESTRATO'],
                        title='Renta vs Área (Tamaño = VT, Color = Similitud)',
                        labels={'AREA': 'Área (m²)', ' RENTA ': 'Renta ($)'},
                        color_continuous_scale='RdYlGn'
                    )
                    
                    st.plotly_chart(fig_renta, use_container_width=True)
                
                with tab2:
                    # Distribución por zona
                    dist_zona = top_10['ZONA'].value_counts()
                    
                    fig_zona = px.pie(
                        values=dist_zona.values,
                        names=dist_zona.index,
                        title='Distribución de Tiendas Espejo por Zona'
                    )
                    
                    st.plotly_chart(fig_zona, use_container_width=True)
                    
                    # Distribución por estrato
                    dist_estrato = top_10['ESTRATO'].value_counts().sort_index()
                    
                    fig_estrato = px.bar(
                        x=dist_estrato.index,
                        y=dist_estrato.values,
                        title='Distribución por Estrato',
                        labels={'x': 'Estrato', 'y': 'Cantidad'},
                        color=dist_estrato.values,
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig_estrato, use_container_width=True)
                
                with tab3:
                    # Gráfico de similitud
                    fig_sim = px.bar(
                        top_10,
                        x='NAME',
                        y='SIMILITUD',
                        title='Porcentaje de Similitud - Top 10',
                        labels={'NAME': 'Tienda', 'SIMILITUD': 'Similitud (%)'},
                        color='SIMILITUD',
                        color_continuous_scale='RdYlGn'
                    )
                    
                    fig_sim.update_layout(
                        xaxis_tickangle=-45,
                        height=400
                    )
                    
                    st.plotly_chart(fig_sim, use_container_width=True)
                    
                    # Tabla resumen de características vs similitud
                    st.markdown("#### 📋 Factores de Similitud")
                    
                    mejor_tienda = resultado.iloc[0]
                    
                    comparacion = pd.DataFrame({
                        'Característica': ['Segmento', 'Zona', 'Municipio', 'Estrato', 'Tipo de Local', 'Generador'],
                        'Tu Propuesta': [
                            nueva_tienda['SEG26'],
                            nueva_tienda['ZONA'],
                            nueva_tienda['MUN'],
                            nueva_tienda['ESTRATO'],
                            nueva_tienda['TIPO DE LOCAL'],
                            nueva_tienda['GENERADOR']
                        ],
                        'Tienda Espejo': [
                            mejor_tienda['SEG26'],
                            mejor_tienda['ZONA'],
                            mejor_tienda['MUN'],
                            mejor_tienda['ESTRATO'],
                            mejor_tienda['TIPO DE LOCAL'],
                            mejor_tienda['GENERADOR']
                        ],
                        'Coincide': [
                            '✅' if nueva_tienda['SEG26'] == mejor_tienda['SEG26'] else '❌',
                            '✅' if nueva_tienda['ZONA'] == mejor_tienda['ZONA'] else '❌',
                            '✅' if nueva_tienda['MUN'] == mejor_tienda['MUN'] else '❌',
                            '✅' if nueva_tienda['ESTRATO'] == mejor_tienda['ESTRATO'] else '❌',
                            '✅' if nueva_tienda['TIPO DE LOCAL'] == mejor_tienda['TIPO DE LOCAL'] else '❌',
                            '✅' if nueva_tienda['GENERADOR'] == mejor_tienda['GENERADOR'] else '❌'
                        ]
                    })
                    
                    st.dataframe(comparacion, use_container_width=True, hide_index=True)

else:
    st.info("👈 Por favor, carga un archivo Excel en la barra lateral para comenzar")
    
    st.markdown("""
    ### 📖 Cómo usar esta herramienta:
    
    1. **Carga tu archivo Excel** en la barra lateral (o usa los datos precargados)
    2. **Completa los datos** de la nueva tienda propuesta
    3. **Ajusta los pesos** de las características según tu criterio
    4. **Haz clic en "Buscar Tienda Espejo"** para ver los resultados
    
    El modelo buscará las tiendas operativas más similares considerando:
    - ✅ Mismo segmento (obligatorio)
    - 📍 Zona geográfica
    - 🏘️ Estrato socioeconómico
    - 🏢 Tipo de local
    - 📏 Área del local
    - ⚡ Tipo de generador
    - 🗺️ Municipio
    
    ### 🎯 El resultado te mostrará:
    - La mejor tienda espejo con su % de similitud
    - Ventas (VT) y tráfico (ET) de referencia
    - Top 10 alternativas
    - Opción de descargar resultados completos
    """)

# Footer
st.divider()
st.caption("🏪 Modelo de Tienda Espejo | Desarrollado con Streamlit")
