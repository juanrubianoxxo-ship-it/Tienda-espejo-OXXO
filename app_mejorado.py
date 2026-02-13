import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
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
st.title("🏪 Modelo de Tienda Espejo v1.0")
st.markdown("### Encuentra la tienda operativa más similar a tu propuesta")

# Función para calcular similitud con modelo estadístico mejorado
def calcular_tienda_espejo_estadistico(df, nueva_tienda, pesos=None):
    """
    Calcula las tiendas más similares usando distancia euclidiana ponderada normalizada.
    
    Metodología:
    1. Filtra por segmento (obligatorio)
    2. Normaliza variables numéricas con StandardScaler
    3. Codifica variables categóricas (1 si coincide, 0 si no)
    4. Calcula distancia euclidiana ponderada
    5. Convierte distancia a score de similitud (0-100%)
    """
    if pesos is None:
        pesos = {
            'SEG26': 0.30,
            'ZONA': 0.12,
            'ESTRATO': 0.10,
            'TIPO DE LOCAL': 0.08,
            'AREA': 0.10,
            'GENERADOR': 0.08,
            'MUN': 0.07,
            'VIVIENDAS': 0.075,
            'EMPLEOS': 0.075
        }
    
    # Filtrar tiendas del mismo segmento (obligatorio)
    df_filtrado = df[df['SEG26'] == nueva_tienda['SEG26']].copy()
    
    if len(df_filtrado) == 0:
        return None, "No se encontraron tiendas en el mismo segmento"
    
    # === PREPARACIÓN DE VARIABLES NUMÉRICAS ===
    vars_numericas = ['ESTRATO', 'AREA', 'VIVIENDAS', 'EMPLEOS']
    
    # Crear matriz de variables numéricas
    X_num_df = df_filtrado[vars_numericas].copy()
    X_num_nueva = np.array([[
        nueva_tienda['ESTRATO'],
        nueva_tienda['AREA'],
        nueva_tienda['VIVIENDAS'],
        nueva_tienda['EMPLEOS']
    ]])
    
    # Normalizar usando StandardScaler (media=0, std=1)
    scaler = StandardScaler()
    X_num_df_scaled = scaler.fit_transform(X_num_df)
    X_num_nueva_scaled = scaler.transform(X_num_nueva)
    
    # === PREPARACIÓN DE VARIABLES CATEGÓRICAS ===
    vars_categoricas = ['ZONA', 'TIPO DE LOCAL', 'GENERADOR', 'MUN']
    
    # Crear matriz de coincidencias categóricas (1 = coincide, 0 = no coincide)
    X_cat_df = np.zeros((len(df_filtrado), len(vars_categoricas)))
    
    for i, var in enumerate(vars_categoricas):
        X_cat_df[:, i] = (df_filtrado[var] == nueva_tienda[var]).astype(int)
    
    X_cat_nueva = np.ones((1, len(vars_categoricas)))  # Nueva tienda coincide consigo misma
    
    # === COMBINAR CARACTERÍSTICAS ===
    # Combinar variables numéricas normalizadas y categóricas
    X_df_completo = np.hstack([X_num_df_scaled, X_cat_df])
    X_nueva_completo = np.hstack([X_num_nueva_scaled, X_cat_nueva])
    
    # === APLICAR PESOS ===
    # Crear vector de pesos para cada característica
    peso_vector = np.array([
        pesos['ESTRATO'],
        pesos['AREA'],
        pesos['VIVIENDAS'],
        pesos['EMPLEOS'],
        pesos['ZONA'],
        pesos['TIPO DE LOCAL'],
        pesos['GENERADOR'],
        pesos['MUN']
    ])
    
    # Ponderar las características multiplicando por raíz cuadrada del peso
    # (para que el peso se aplique en la distancia euclidiana)
    X_df_ponderado = X_df_completo * np.sqrt(peso_vector)
    X_nueva_ponderado = X_nueva_completo * np.sqrt(peso_vector)
    
    # === CALCULAR DISTANCIA EUCLIDIANA ===
    distancias = euclidean_distances(X_nueva_ponderado, X_df_ponderado)[0]
    
    # === CONVERTIR DISTANCIA A SCORE DE SIMILITUD ===
    # Normalizar distancias a rango 0-100
    # Usando transformación exponencial inversa para mejor interpretación
    max_dist = np.max(distancias)
    min_dist = np.min(distancias)
    
    if max_dist > min_dist:
        # Normalizar a 0-1 y luego invertir (menor distancia = mayor similitud)
        distancias_norm = (distancias - min_dist) / (max_dist - min_dist)
        similitud_scores = (1 - distancias_norm) * 100
    else:
        # Si todas las distancias son iguales
        similitud_scores = np.full_like(distancias, 100.0)
    
    # Añadir scores al dataframe
    df_resultado = df_filtrado.copy()
    df_resultado['DISTANCIA'] = distancias
    df_resultado['SIMILITUD'] = similitud_scores
    
    # Ordenar por similitud (mayor a menor)
    df_resultado = df_resultado.sort_values('SIMILITUD', ascending=False)
    
    return df_resultado, None


# Función para calcular estadísticas descriptivas
def calcular_estadisticas(df_resultado, nueva_tienda):
    """
    Calcula estadísticas descriptivas del top de tiendas espejo
    Nota: VT = Viviendas Totales, ET = Empleos Totales
    """
    top_10 = df_resultado.head(10)
    
    # Buscar la columna de renta
    renta_col = None
    for col in df_resultado.columns:
        if 'RENTA' in col.upper():
            renta_col = col
            break
    
    # Buscar columnas de ventas y tráfico (si existen con otros nombres)
    ventas_col = None
    trafico_col = None
    for col in df_resultado.columns:
        if 'VENTA' in col.upper() or col.upper() == 'SALES':
            ventas_col = col
        if 'TRAFICO' in col.upper() or 'TRÁFICO' in col.upper() or col.upper() == 'TRAFFIC':
            trafico_col = col
    
    # Si no se encuentra, usar un valor por defecto
    if renta_col is None or renta_col not in df_resultado.columns:
        stats = {
            'VT_promedio': top_10['VT'].mean(),  # VT = Viviendas
            'VT_std': top_10['VT'].std(),
            'VT_min': top_10['VT'].min(),
            'VT_max': top_10['VT'].max(),
            'ET_promedio': top_10['ET'].mean(),  # ET = Empleos
            'ET_std': top_10['ET'].std(),
            'RENTA_promedio': 0,
            'RENTA_std': 0,
            'AREA_promedio': top_10['AREA'].mean(),
            'similitud_promedio': top_10['SIMILITUD'].mean(),
            'renta_col': 'RENTA',
            'ventas_col': ventas_col,
            'trafico_col': trafico_col
        }
        return stats
    
    stats = {
        'VT_promedio': top_10['VT'].mean(),  # VT = Viviendas
        'VT_std': top_10['VT'].std(),
        'VT_min': top_10['VT'].min(),
        'VT_max': top_10['VT'].max(),
        'ET_promedio': top_10['ET'].mean(),  # ET = Empleos
        'ET_std': top_10['ET'].std(),
        'RENTA_promedio': top_10[renta_col].mean(),
        'RENTA_std': top_10[renta_col].std(),
        'AREA_promedio': top_10['AREA'].mean(),
        'similitud_promedio': top_10['SIMILITUD'].mean(),
        'renta_col': renta_col,
        'ventas_col': ventas_col,
        'trafico_col': trafico_col
    }
    
    return stats


# Sidebar para cargar datos
with st.sidebar:
    st.header("📂 Cargar Datos")
    
    # Opción para usar archivo de ejemplo o cargar uno nuevo
    usar_ejemplo = st.checkbox("Usar datos precargados", value=True)
    
    if usar_ejemplo:
        try:
            df = pd.read_excel('Book.xlsx')
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
        st.info("💡 Los pesos se normalizan automáticamente para sumar 100%")
        
        peso_seg = st.slider("Segmento (filtro obligatorio)", 0, 100, 30, disabled=True, 
                            help="Las tiendas se filtran obligatoriamente por el mismo segmento")
        peso_zona = st.slider("Zona geográfica", 0, 100, 12)
        peso_estrato = st.slider("Estrato socioeconómico", 0, 100, 10)
        peso_tipo = st.slider("Tipo de Local", 0, 100, 8)
        peso_area = st.slider("Área (m²)", 0, 100, 10)
        peso_generador = st.slider("Generador", 0, 100, 8)
        peso_mun = st.slider("Municipio", 0, 100, 7)
        peso_viviendas = st.slider("Viviendas Totales", 0, 100, 7)
        peso_empleos = st.slider("Empleos Totales", 0, 100, 8)
        
        # Normalizar pesos (excluir segmento que es 30% fijo)
        total = peso_zona + peso_estrato + peso_tipo + peso_area + peso_generador + peso_mun + peso_viviendas + peso_empleos
        
        if total > 0:
            pesos = {
                'SEG26': 0.30,  # Fijo - 30%
                'ZONA': peso_zona / total * 0.70,
                'ESTRATO': peso_estrato / total * 0.70,
                'TIPO DE LOCAL': peso_tipo / total * 0.70,
                'AREA': peso_area / total * 0.70,
                'GENERADOR': peso_generador / total * 0.70,
                'MUN': peso_mun / total * 0.70,
                'VIVIENDAS': peso_viviendas / total * 0.70,
                'EMPLEOS': peso_empleos / total * 0.70
            }
        else:
            pesos = None
        
        # Mostrar pesos finales
        with st.expander("Ver pesos normalizados"):
            if pesos:
                for key, val in pesos.items():
                    st.write(f"**{key}:** {val*100:.1f}%")

# Contenido principal
if df is not None:
    # Verificar que las columnas necesarias existan
    columnas_requeridas = ['SEG26', 'ZONA', 'MUN', 'ESTRATO', 'TIPO DE LOCAL', 
                           'AREA', 'GENERADOR', 'VT', 'ET', 'CR', 'NAME']
    
    # VT = Viviendas Totales, ET = Empleos Totales
    # Crear alias para mantener compatibilidad con el código
    df['VIVIENDAS'] = df['VT']
    df['EMPLEOS'] = df['ET']
    
    # Verificar si existe columna de RENTA
    renta_col_disponible = None
    for col in df.columns:
        if 'RENTA' in col.upper():
            renta_col_disponible = col
            break
    
    if renta_col_disponible is None:
        st.warning("⚠️ No se encontró columna de RENTA en el dataset.")
        df['RENTA'] = 0
        renta_col_disponible = 'RENTA'
    
    # Dos columnas: entrada de datos y resultados
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Nueva Tienda Propuesta")
        
        # Formulario para ingresar datos de la nueva tienda
        with st.form("form_nueva_tienda"):
            nombre_nueva = st.text_input("Nombre de la tienda propuesta", "Mi Nueva Tienda")
            
            st.markdown("##### Características Principales")
            segmento = st.selectbox("Segmento (SEG26)", options=sorted(df['SEG26'].unique()))
            zona = st.selectbox("Zona", options=sorted(df['ZONA'].unique()))
            municipio = st.selectbox("Municipio", options=sorted(df['MUN'].unique()))
            estrato = st.selectbox("Estrato", options=sorted(df['ESTRATO'].unique()))
            tipo_local = st.selectbox("Tipo de Local", options=sorted(df['TIPO DE LOCAL'].unique()))
            generador = st.selectbox("Generador", options=sorted(df['GENERADOR'].unique()))
            
            st.markdown("##### Métricas Numéricas")
            col_a, col_b = st.columns(2)
            with col_a:
                area = st.number_input("Área (m²)", min_value=0.0, value=100.0, step=10.0)
                viviendas = st.number_input("Viviendas Totales", min_value=0, value=1000, step=100,
                                           help="Número total de viviendas en el área de influencia")
            with col_b:
                empleos = st.number_input("Empleos Totales", min_value=0, value=500, step=50,
                                         help="Número total de empleos en el área de influencia")
            
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
                'GENERADOR': generador,
                'VIVIENDAS': viviendas,
                'EMPLEOS': empleos
            }
            
            # Calcular tiendas espejo con modelo estadístico
            resultado, error = calcular_tienda_espejo_estadistico(df, nueva_tienda, pesos)
            
            if error:
                st.error(error)
            else:
                # Calcular estadísticas
                stats = calcular_estadisticas(resultado, nueva_tienda)
                renta_col = stats['renta_col']
                
                # Mostrar top 5
                st.success("✅ Tiendas espejo encontradas usando modelo estadístico")
                
                # Tarjeta de la mejor tienda espejo
                mejor = resultado.iloc[0]
                
                st.markdown("### 🏆 Mejor Tienda Espejo")
                
                # Crear tres columnas para info destacada
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Nombre", mejor['NAME'])
                    st.metric("Código", mejor['CR'])
                with c2:
                    st.metric("Similitud", f"{mejor['SIMILITUD']:.1f}%")
                    st.metric("Distancia", f"{mejor['DISTANCIA']:.3f}")
                with c3:
                    st.metric("Viviendas (VT)", f"{mejor['VT']:,.0f}")
                    st.metric("Empleos (ET)", f"{mejor['ET']:,.0f}")
                with c4:
                    if renta_col in mejor.index and mejor[renta_col] > 0:
                        st.metric("Renta", f"${mejor[renta_col]:,.0f}")
                    else:
                        st.metric("Renta", "N/A")
                    st.metric("Área", f"{mejor['AREA']:.1f} m²")
                
                # Detalles de la mejor tienda
                with st.expander("📊 Ver detalles completos de la mejor tienda", expanded=False):
                    col_det1, col_det2 = st.columns(2)
                    with col_det1:
                        st.write(f"**Segmento:** {mejor['SEG26']}")
                        st.write(f"**Zona:** {mejor['ZONA']}")
                        st.write(f"**Municipio:** {mejor['MUN']}")
                        st.write(f"**Estrato:** {mejor['ESTRATO']}")
                    with col_det2:
                        st.write(f"**Tipo de Local:** {mejor['TIPO DE LOCAL']}")
                        st.write(f"**Generador:** {mejor['GENERADOR']}")
                        st.write(f"**Viviendas (VT):** {mejor['VT']:,.0f}")
                        st.write(f"**Empleos (ET):** {mejor['ET']:,.0f}")
                
                st.divider()
                
                # Estadísticas del Top 10
                st.markdown("### 📈 Estadísticas del Top 10")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    st.metric("Viviendas Promedio (VT)", f"{stats['VT_promedio']:,.0f}")
                    st.caption(f"±{stats['VT_std']:,.0f}")
                
                with col_stat2:
                    st.metric("Empleos Promedio (ET)", f"{stats['ET_promedio']:,.0f}")
                    st.caption(f"±{stats['ET_std']:,.0f}")
                
                with col_stat3:
                    if stats['RENTA_promedio'] > 0:
                        st.metric("Renta Promedio", f"${stats['RENTA_promedio']:,.0f}")
                        st.caption(f"±{stats['RENTA_std']:,.0f}")
                    else:
                        st.metric("Renta Promedio", "N/A")
                
                with col_stat4:
                    st.metric("Similitud Promedio", f"{stats['similitud_promedio']:.1f}%")
                    st.caption(f"Área: {stats['AREA_promedio']:.1f} m²")
                
                st.divider()
                
                # Top 10 alternativas
                st.markdown("### 📋 Top 10 Alternativas")
                
                # Preparar dataframe para mostrar
                columnas_mostrar = ['CR', 'NAME', 'ZONA', 'MUN', 'ESTRATO', 
                                   'TIPO DE LOCAL', 'AREA', 'VT', 'ET']
                
                if renta_col in resultado.columns:
                    columnas_mostrar.append(renta_col)
                
                columnas_mostrar.extend(['SIMILITUD', 'DISTANCIA'])
                
                top_10 = resultado.head(10)[columnas_mostrar]
                
                # Formatear columnas
                top_10_display = top_10.copy()
                top_10_display['SIMILITUD'] = top_10_display['SIMILITUD'].apply(lambda x: f"{x:.1f}%")
                top_10_display['DISTANCIA'] = top_10_display['DISTANCIA'].apply(lambda x: f"{x:.3f}")
                top_10_display['AREA'] = top_10_display['AREA'].apply(lambda x: f"{x:.1f}")
                top_10_display['VT'] = top_10_display['VT'].apply(lambda x: f"{x:,.0f}")
                top_10_display['ET'] = top_10_display['ET'].apply(lambda x: f"{x:,.0f}")
                if renta_col in top_10_display.columns:
                    top_10_display[renta_col] = top_10_display[renta_col].apply(lambda x: f"${x:,.0f}")
                
                # Renombrar columnas para mayor claridad
                top_10_display = top_10_display.rename(columns={
                    'VT': 'Viviendas (VT)',
                    'ET': 'Empleos (ET)'
                })
                
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
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Comparación de Métricas", 
                    "Distribución Geográfica", 
                    "Análisis de Similitud",
                    "Modelo Estadístico"
                ])
                
                with tab1:
                    # Comparación de métricas principales
                    top_5 = resultado.head(5)
                    
                    fig_metricas = go.Figure()
                    
                    # VT (Viviendas)
                    fig_metricas.add_trace(go.Bar(
                        name='Viviendas (VT)',
                        x=top_5['NAME'],
                        y=top_5['VT'],
                        marker_color='lightblue'
                    ))
                    
                    # ET (Empleos)
                    fig_metricas.add_trace(go.Bar(
                        name='Empleos (ET)',
                        x=top_5['NAME'],
                        y=top_5['ET'],
                        marker_color='lightgreen'
                    ))
                    
                    fig_metricas.update_layout(
                        title='Top 5 Tiendas Espejo - Viviendas vs Empleos',
                        xaxis_title='Tienda',
                        yaxis_title='Cantidad',
                        barmode='group',
                        height=400
                    )
                    
                    st.plotly_chart(fig_metricas, use_container_width=True)
                    
                    # Gráfico de renta vs área (solo si hay renta)
                    if renta_col in top_10.columns and top_10[renta_col].sum() > 0:
                        fig_renta = px.scatter(
                            top_10,
                            x='AREA',
                            y=renta_col,
                            size='VT',
                            color='SIMILITUD',
                            hover_data=['NAME', 'ZONA', 'ESTRATO', 'VT', 'ET'],
                            title='Renta vs Área (Tamaño = Viviendas VT, Color = Similitud)',
                            labels={'AREA': 'Área (m²)', renta_col: 'Renta ($)', 'VT': 'Viviendas', 'ET': 'Empleos'},
                            color_continuous_scale='RdYlGn'
                        )
                        
                        st.plotly_chart(fig_renta, use_container_width=True)
                    
                    # Gráfico VT (Viviendas) vs ET (Empleos)
                    fig_viv_emp = px.scatter(
                        top_10,
                        x='VT',
                        y='ET',
                        size='AREA',
                        color='SIMILITUD',
                        hover_data=['NAME', 'ZONA'],
                        title='Viviendas (VT) vs Empleos (ET) - Tamaño = Área',
                        labels={'VT': 'Viviendas Totales (VT)', 'ET': 'Empleos Totales (ET)'},
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig_viv_emp, use_container_width=True)
                
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
                        'Característica': ['Segmento', 'Zona', 'Municipio', 'Estrato', 
                                          'Tipo de Local', 'Generador', 'Área', 'Viviendas (VT)', 'Empleos (ET)'],
                        'Tu Propuesta': [
                            nueva_tienda['SEG26'],
                            nueva_tienda['ZONA'],
                            nueva_tienda['MUN'],
                            nueva_tienda['ESTRATO'],
                            nueva_tienda['TIPO DE LOCAL'],
                            nueva_tienda['GENERADOR'],
                            f"{nueva_tienda['AREA']:.1f} m²",
                            f"{nueva_tienda['VIVIENDAS']:,}",
                            f"{nueva_tienda['EMPLEOS']:,}"
                        ],
                        'Tienda Espejo': [
                            mejor_tienda['SEG26'],
                            mejor_tienda['ZONA'],
                            mejor_tienda['MUN'],
                            mejor_tienda['ESTRATO'],
                            mejor_tienda['TIPO DE LOCAL'],
                            mejor_tienda['GENERADOR'],
                            f"{mejor_tienda['AREA']:.1f} m²",
                            f"{mejor_tienda['VT']:,.0f}",
                            f"{mejor_tienda['ET']:,.0f}"
                        ],
                        'Coincide': [
                            '✅' if nueva_tienda['SEG26'] == mejor_tienda['SEG26'] else '❌',
                            '✅' if nueva_tienda['ZONA'] == mejor_tienda['ZONA'] else '❌',
                            '✅' if nueva_tienda['MUN'] == mejor_tienda['MUN'] else '❌',
                            '✅' if nueva_tienda['ESTRATO'] == mejor_tienda['ESTRATO'] else '❌',
                            '✅' if nueva_tienda['TIPO DE LOCAL'] == mejor_tienda['TIPO DE LOCAL'] else '❌',
                            '✅' if nueva_tienda['GENERADOR'] == mejor_tienda['GENERADOR'] else '❌',
                            f"{abs(nueva_tienda['AREA'] - mejor_tienda['AREA']):.1f} m²",
                            f"{abs(nueva_tienda['VIVIENDAS'] - mejor_tienda['VT']):,.0f}",
                            f"{abs(nueva_tienda['EMPLEOS'] - mejor_tienda['ET']):,.0f}"
                        ]
                    })
                    
                    st.dataframe(comparacion, use_container_width=True, hide_index=True)
                
                with tab4:
                    st.markdown("#### 🔬 Modelo Estadístico: Distancia Euclidiana Ponderada")
                    
                    st.markdown("""
                    **Metodología del modelo:**
                    
                    1. **Filtrado**: Se filtran tiendas del mismo segmento (SEG26)
                    2. **Normalización**: Variables numéricas se normalizan usando StandardScaler (μ=0, σ=1)
                    3. **Codificación**: Variables categóricas se codifican binariamente (1=coincide, 0=no coincide)
                    4. **Ponderación**: Se aplican pesos configurables a cada variable
                    5. **Distancia**: Se calcula distancia euclidiana ponderada en espacio multidimensional
                    6. **Similitud**: La distancia se invierte y normaliza a escala 0-100%
                    
                    **Variables incluidas:**
                    - Numéricas normalizadas: ESTRATO, ÁREA, VIVIENDAS, EMPLEOS
                    - Categóricas: ZONA, TIPO DE LOCAL, GENERADOR, MUNICIPIO
                    
                    **Ventajas:**
                    - Modelo estadístico robusto y reproducible
                    - Considera todas las dimensiones simultáneamente
                    - Pesos configurables según criterio de negocio
                    - Normalización evita que variables con rangos grandes dominen
                    """)
                    
                    # Distribución de distancias
                    fig_dist = px.histogram(
                        resultado.head(50),
                        x='DISTANCIA',
                        nbins=20,
                        title='Distribución de Distancias Euclidianas (Top 50)',
                        labels={'DISTANCIA': 'Distancia Euclidiana', 'count': 'Frecuencia'},
                        color_discrete_sequence=['indianred']
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Relación Distancia vs Similitud
                    fig_rel = px.scatter(
                        resultado.head(30),
                        x='DISTANCIA',
                        y='SIMILITUD',
                        hover_data=['NAME'],
                        title='Relación Distancia vs Similitud (Top 30)',
                        labels={'DISTANCIA': 'Distancia Euclidiana', 'SIMILITUD': 'Similitud (%)'},
                        trendline="ols"
                    )
                    
                    st.plotly_chart(fig_rel, use_container_width=True)

else:
    st.info("👈 Por favor, carga un archivo Excel en la barra lateral para comenzar")
    
    st.markdown("""
    ### 📖 Cómo usar esta herramienta:
    
    1. **Carga tu archivo Excel** en la barra lateral (o usa los datos precargados)
    2. **Completa los datos** de la nueva tienda propuesta, incluyendo:
       - Características principales (segmento, zona, municipio, etc.)
       - **Viviendas Totales** en el área de influencia
       - **Empleos Totales** en el área de influencia
    3. **Ajusta los pesos** de las características según tu criterio
    4. **Haz clic en "Buscar Tienda Espejo"** para ver los resultados
    
    ### 🎯 Modelo Estadístico Robusto:
    
    Esta versión mejorada utiliza **distancia euclidiana ponderada normalizada** que:
    - ✅ Normaliza todas las variables numéricas (evita sesgos por escalas diferentes)
    - ✅ Codifica variables categóricas de forma binaria
    - ✅ Aplica pesos configurables a cada dimensión
    - ✅ Calcula similitud en espacio multidimensional
    - ✅ Incluye **Viviendas (VT)** y **Empleos (ET)** como variables clave
    
    ### 📊 Variables consideradas:
    - 🏢 **Segmento** (filtro obligatorio - 30% peso fijo)
    - 📍 **Zona geográfica**
    - 🏘️ **Estrato socioeconómico**
    - 🏪 **Tipo de local**
    - 📏 **Área del local**
    - ⚡ **Tipo de generador**
    - 🗺️ **Municipio**
    - 🏠 **Viviendas Totales (VT)**
    - 💼 **Empleos Totales (ET)**
    
    ### 🎯 El resultado te mostrará:
    - La mejor tienda espejo con % de similitud y distancia euclidiana
    - Estadísticas descriptivas del Top 10 (promedios, desviaciones)
    - Viviendas (VT) y Empleos (ET) de referencia
    - Top 10 alternativas con métricas completas
    - Visualizaciones interactivas del modelo estadístico
    - Opción de descargar resultados completos
    
    ### 📝 Nota sobre las columnas:
    - **VT** = Viviendas Totales en el área de influencia
    - **ET** = Empleos Totales en el área de influencia
    """)

# Footer
st.divider()
st.caption("🏪 Modelo de Tienda Espejo v2.0")
