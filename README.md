# 🏪 Modelo de Tienda Espejo - Guía de Instalación

## 📋 Descripción
Esta herramienta encuentra automáticamente la "tienda espejo" más similar a una nueva propuesta, basándose en múltiples características como segmento, zona, estrato, área, etc.

## 🚀 Opción 1: Desplegar en Streamlit Cloud (GRATIS y RECOMENDADO)

### Paso 1: Crear cuenta en GitHub
1. Ve a https://github.com
2. Crea una cuenta gratuita si no tienes una

### Paso 2: Subir los archivos
1. Crea un nuevo repositorio en GitHub (botón "New repository")
2. Nómbralo como quieras, ej: "tienda-espejo"
3. Márcalo como "Public"
4. Sube estos archivos:
   - `app.py`
   - `requirements.txt`
   - `Book.xlsx` (tu archivo de datos)

### Paso 3: Desplegar en Streamlit Cloud
1. Ve a https://share.streamlit.io
2. Haz clic en "Sign up" y usa tu cuenta de GitHub
3. Haz clic en "New app"
4. Selecciona:
   - Repository: tu repositorio (ej: "tienda-espejo")
   - Branch: main
   - Main file path: app.py
5. Haz clic en "Deploy"

¡Listo! En 2-3 minutos tendrás tu app en línea con una URL que puedes compartir.

### Ventajas de Streamlit Cloud:
✅ Completamente GRATIS
✅ URL pública para compartir
✅ Se actualiza automáticamente cuando subes cambios
✅ No requiere conocimientos de servidores
✅ Disponible 24/7

---

## 🖥️ Opción 2: Ejecutar Localmente

### Requisitos:
- Python 3.8 o superior

### Instalación:

1. Abre la terminal/cmd en la carpeta con los archivos

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación:
```bash
streamlit run app.py
```

4. Se abrirá automáticamente en tu navegador en: http://localhost:8501

---

## 📖 Cómo Usar la Herramienta

### 1. Cargar Datos
- Usa los datos precargados (Book.xlsx)
- O sube tu propio archivo Excel con las columnas necesarias

### 2. Configurar Pesos (Opcional)
En la barra lateral puedes ajustar la importancia de cada característica:
- **Zona**: Importancia de la ubicación geográfica
- **Estrato**: Importancia del nivel socioeconómico
- **Tipo de Local**: Importancia del formato del local
- **Área**: Importancia del tamaño
- **Generador**: Importancia del tipo de generador
- **Municipio**: Importancia del municipio

### 3. Ingresar Nueva Tienda
Completa el formulario con los datos de la tienda propuesta:
- Nombre
- Segmento (obligatorio - debe coincidir)
- Zona
- Municipio
- Estrato
- Tipo de Local
- Área en m²
- Generador

### 4. Ver Resultados
La herramienta te mostrará:
- 🏆 **Mejor Tienda Espejo**: La más similar con % de similitud
- 📊 **Características**: Detalles completos de la tienda espejo
- 📈 **Métricas**: VT (ventas), ET (tráfico), Renta
- 📋 **Top 10**: Las 10 tiendas más similares
- 📥 **Descarga**: Opción de descargar Top 20 en CSV

---

## 🔧 Estructura de Datos Requerida

Tu archivo Excel debe tener estas columnas:
- `CR`: Código de la tienda
- `NAME`: Nombre de la tienda
- `ZONA`: Zona geográfica
- `MUN`: Municipio
- `ESTRATO`: Estrato socioeconómico (número)
- `TIPO DE LOCAL`: Tipo de local
- `AREA`: Área en m² (número)
- `SEG26`: Segmento
- `RENTA`: Renta mensual (número)
- `GENERADOR`: Tipo de generador
- `VT`: Ventas totales (número)
- `ET`: Tráfico estimado (número)

---

## 🎯 Algoritmo de Similitud

El modelo calcula la similitud usando:

1. **Filtro obligatorio**: Mismo segmento (SEG26)
2. **Variables categóricas**: Coincidencia exacta en zona, tipo de local, generador, municipio
3. **Variables numéricas**: Similitud basada en diferencia relativa
4. **Ponderación**: Score final basado en los pesos configurados
5. **Ranking**: Ordenamiento por % de similitud (0-100%)

### Fórmula de Similitud:
```
Similitud = Σ (Score_característica × Peso_característica)

Donde:
- Score_característica = 1 (coincide) o 0 (no coincide) para categóricas
- Score_característica = 1 / (1 + diferencia_relativa) para numéricas
```

---

## 💡 Consejos de Uso

1. **Segmento es clave**: El modelo solo busca en tiendas del mismo segmento
2. **Ajusta pesos**: Según tu estrategia, dale más peso a las características más importantes
3. **Revisa Top 10**: A veces la segunda o tercera opción puede ser mejor
4. **Compara métricas**: Usa VT y ET como referencia de potencial de la nueva tienda
5. **Descarga resultados**: Guarda el CSV para análisis más profundo

---

## 🆘 Solución de Problemas

### Error al cargar archivo
- Verifica que sea formato .xlsx
- Asegúrate de que tenga todas las columnas requeridas
- Revisa que no haya espacios extras en nombres de columnas

### No encuentra tiendas espejo
- Verifica que existan tiendas en ese segmento
- Intenta cambiar los filtros o pesos

### La app no carga
- Espera 2-3 minutos después del deploy
- Refresca la página
- Verifica que todos los archivos estén en GitHub

---

## 📧 Soporte

Si encuentras algún problema o tienes sugerencias:
1. Revisa la documentación arriba
2. Verifica los logs en Streamlit Cloud
3. Ajusta los pesos y parámetros según tu necesidad

---

## 🔄 Actualizar la Aplicación

Para actualizar la app en Streamlit Cloud:
1. Modifica los archivos localmente
2. Sube los cambios a GitHub
3. Streamlit Cloud se actualizará automáticamente en 1-2 minutos

---

## 📊 Ejemplo de Uso

**Caso**: Quieres abrir una tienda en Bogotá, zona Norte, estrato 4, de 120m²

1. Selecciona el segmento de tu interés
2. Configura los demás campos
3. El modelo te dirá: "La tienda espejo es '**Unicentro**' con 92% de similitud"
4. Puedes usar las métricas de Unicentro (VT, ET, Renta) como referencia
5. Revisa las otras 9 alternativas para validar

---

¡Tu herramienta está lista para usar! 🎉
