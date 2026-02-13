# Modelo Estadístico de Tienda Espejo - Documentación Técnica

## 📊 Resumen Ejecutivo

Se ha implementado un modelo estadístico robusto basado en **distancia euclidiana ponderada normalizada** para encontrar tiendas espejo (similares) a una nueva propuesta comercial.

---

## 🎯 Objetivo del Modelo

Identificar las tiendas operativas más similares a una nueva propuesta basándose en múltiples características demográficas, geográficas y comerciales.

---

## 📐 Metodología Estadística

### 1. Filtrado Inicial
- **Filtro obligatorio**: Segmento SEG26 (mismo segmento comercial)
- Reduce el espacio de búsqueda a tiendas comparables

### 2. Normalización de Variables Numéricas

Se utiliza **StandardScaler** (normalización Z-score):

```
X_normalizado = (X - μ) / σ
```

Donde:
- μ = media de la variable
- σ = desviación estándar

**Variables normalizadas:**
- ESTRATO (1-6)
- ÁREA (m²)
- VIVIENDAS TOTALES
- EMPLEOS TOTALES

**Ventaja**: Evita que variables con rangos grandes (ej: viviendas en miles) dominen sobre variables con rangos pequeños (ej: estrato 1-6).

### 3. Codificación de Variables Categóricas

Variables categóricas se codifican binariamente:
- 1 = La característica coincide con la propuesta
- 0 = La característica no coincide

**Variables categóricas:**
- ZONA
- TIPO DE LOCAL
- GENERADOR
- MUNICIPIO

### 4. Ponderación

Cada característica recibe un peso configurable (w₁, w₂, ..., wₙ).

**Distribución de pesos por defecto:**
- Segmento: 30% (fijo)
- Zona: 12%
- Estrato: 10%
- Área: 10%
- Viviendas: 7.5%
- Empleos: 7.5%
- Tipo de Local: 8%
- Generador: 8%
- Municipio: 7%

**Total: 100%**

Los pesos se aplican mediante multiplicación por √w para que afecten la distancia euclidiana correctamente.

### 5. Cálculo de Distancia Euclidiana Ponderada

Para cada tienda i en el dataset:

```
d(nueva, tienda_i) = √(Σ w_j * (x_j_nueva - x_j_i)²)
```

Donde:
- d = distancia euclidiana
- w_j = peso de la característica j
- x_j = valor normalizado de la característica j

### 6. Conversión a Score de Similitud

La distancia se invierte y normaliza a escala 0-100%:

```
Similitud = (1 - (d - d_min) / (d_max - d_min)) × 100
```

**Interpretación:**
- 100% = Coincidencia perfecta (distancia mínima)
- 0% = Máxima disimilitud (distancia máxima)

---

## 🔍 Ventajas del Modelo

1. **Robusto estadísticamente**: Basado en distancia euclidiana, una métrica ampliamente validada
2. **Escalable**: Normalización permite comparar variables en diferentes escalas
3. **Configurable**: Pesos ajustables según criterios de negocio
4. **Multidimensional**: Considera todas las características simultáneamente
5. **Interpretable**: Score de similitud en escala 0-100% fácil de entender
6. **Reproducible**: Mismo input genera mismo output

---

## 📊 Variables Incluidas

### Variables Numéricas (normalizadas)
| Variable | Tipo | Descripción | Rango típico |
|----------|------|-------------|--------------|
| ESTRATO | Ordinal | Estrato socioeconómico | 1-6 |
| ÁREA | Continua | Área del local en m² | 50-500 |
| VIVIENDAS | Discreta | Viviendas en área de influencia | 500-10,000 |
| EMPLEOS | Discreta | Empleos en área de influencia | 100-5,000 |

### Variables Categóricas (binarias)
| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| ZONA | Nominal | Zona geográfica | Norte, Sur, Este, Oeste, etc. |
| TIPO DE LOCAL | Nominal | Tipo de establecimiento | Local comercial, CC, etc. |
| GENERADOR | Nominal | Tipo de generador de tráfico | Ancla, Satélite, etc. |
| MUNICIPIO | Nominal | Municipio de ubicación | Bogotá, Medellín, etc. |

---

## 🎯 Outputs del Modelo

### Métricas por Tienda
- **DISTANCIA**: Distancia euclidiana en espacio normalizado
- **SIMILITUD**: Score 0-100% (mayor = más similar)

### Estadísticas Agregadas (Top 10)
- VT promedio ± desviación estándar
- ET promedio ± desviación estándar  
- Renta promedio ± desviación estándar
- Similitud promedio
- Área promedio

---

## 💡 Casos de Uso

### Alta Similitud (>90%)
- Tiendas prácticamente idénticas
- Excelente referencia para proyecciones de VT/ET
- Renta y costos directamente aplicables

### Similitud Media (70-90%)
- Tiendas comparables con algunas diferencias
- Buena referencia con ajustes menores
- Analizar qué características difieren

### Baja Similitud (<70%)
- Diferencias significativas
- Usar con precaución para proyecciones
- Considerar múltiples referencias

---

## 📈 Ejemplo de Cálculo

**Propuesta nueva:**
- ESTRATO: 4
- ÁREA: 120 m²
- VIVIENDAS: 2,500
- EMPLEOS: 1,200
- ZONA: Norte
- TIPO: Local comercial

**Tienda Candidata:**
- ESTRATO: 4 → diferencia = 0 → similaridad alta
- ÁREA: 115 m² → diferencia pequeña → similaridad alta
- VIVIENDAS: 2,800 → diferencia moderada → similaridad media
- EMPLEOS: 1,000 → diferencia moderada → similaridad media
- ZONA: Norte → coincide → +1
- TIPO: Local comercial → coincide → +1

**Resultado:** Alta similitud por coincidencias categóricas y proximidad numérica.

---

## 🛠️ Configuración Recomendada de Pesos

### Para Retail de Proximidad
- Mayor peso en VIVIENDAS y ZONA
- Menor peso en EMPLEOS

### Para Retail de Destino
- Mayor peso en EMPLEOS y GENERADOR
- Menor peso en VIVIENDAS

### Para Ubicaciones Premium
- Mayor peso en ESTRATO y ZONA
- Menor peso en ÁREA

---

## 📚 Referencias Técnicas

- **StandardScaler**: Scikit-learn preprocessing
- **Distancia Euclidiana**: `scipy.spatial.distance.euclidean`
- **Normalización**: Z-score (μ=0, σ=1)
- **Espacio métrico**: Euclidiano multidimensional

---

## 🔄 Mejoras Futuras Posibles

1. Implementar clustering (K-means) para identificar grupos de tiendas
2. Agregar análisis de componentes principales (PCA)
3. Incorporar modelos predictivos de VT/ET basados en similitud
4. Validación cruzada del modelo con datos históricos
5. Ajuste automático de pesos mediante optimización

---

**Versión:** 2.0  
**Última actualización:** Febrero 2025  
**Autor:** Sistema de Tienda Espejo
