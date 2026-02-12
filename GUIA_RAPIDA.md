# 🚀 GUÍA RÁPIDA - Desplegar en 5 Minutos

## ✅ OPCIÓN RECOMENDADA: Streamlit Cloud (100% GRATIS)

### 📦 Paso 1: Preparar archivos (Ya los tienes listos)
- ✓ app.py
- ✓ requirements.txt  
- ✓ Book.xlsx
- ✓ README.md

### 🌐 Paso 2: Crear cuenta GitHub (2 minutos)
1. Ir a: https://github.com/signup
2. Crear cuenta con tu email
3. Verificar email

### 📤 Paso 3: Subir archivos a GitHub (3 minutos)

#### Opción A - Interfaz Web (Más fácil):
1. Iniciar sesión en GitHub
2. Clic en "+" arriba a la derecha → "New repository"
3. Nombre del repo: `tienda-espejo` (o el que quieras)
4. Marcar "Public"
5. ✓ "Add a README file"
6. Clic "Create repository"
7. En la página del repo, clic "Add file" → "Upload files"
8. Arrastrar TODOS estos archivos:
   - app.py
   - requirements.txt
   - Book.xlsx
   - README.md
9. Clic "Commit changes"

#### Opción B - GitHub Desktop (Alternativa):
1. Descargar GitHub Desktop: https://desktop.github.com
2. Instalar y hacer login
3. "File" → "New Repository"
4. Copiar los archivos a la carpeta del repo
5. "Commit to main" → "Publish repository"

### 🎨 Paso 4: Desplegar en Streamlit Cloud (2 minutos)

1. Ir a: https://share.streamlit.io
2. Clic "Continue with GitHub"
3. Autorizar Streamlit
4. Clic "New app"
5. Completar:
   ```
   Repository: tu-usuario/tienda-espejo
   Branch: main
   Main file path: app.py
   ```
6. Clic "Deploy"
7. ⏳ Esperar 2-3 minutos

### 🎉 ¡LISTO!

Tu app estará disponible en una URL como:
```
https://tu-usuario-tienda-espejo.streamlit.app
```

Puedes compartir esta URL con tu equipo.

---

## 🖥️ ALTERNATIVA: Ejecutar Localmente

Si prefieres probarla primero en tu computadora:

### Windows:
```cmd
pip install -r requirements.txt
streamlit run app.py
```

### Mac/Linux:
```bash
pip3 install -r requirements.txt
streamlit run app.py
```

Se abrirá en: http://localhost:8501

---

## 📱 Cómo Usar la App

### 1️⃣ Configurar (Opcional)
En la barra lateral, ajusta los pesos según importancia:
- Zona: 15%
- Estrato: 15%
- Tipo de Local: 10%
- Área: 10%
- Generador: 10%
- Municipio: 10%

### 2️⃣ Ingresar Datos
Completa el formulario:
- Nombre de la tienda propuesta
- **Segmento** (debe existir en la base de datos)
- Zona, Municipio, Estrato
- Tipo de Local
- Área en m²
- Generador

### 3️⃣ Buscar
Clic en "🔍 Buscar Tienda Espejo"

### 4️⃣ Analizar Resultados
La app te muestra:

📊 **Mejor Tienda Espejo:**
- Nombre y código
- % de similitud
- VT (ventas), ET (tráfico), Renta
- Todas sus características

📋 **Top 10 Alternativas:**
- Tabla completa con todas las métricas
- Ordenadas por similitud

📈 **Visualizaciones:**
- Comparación de métricas (VT vs ET)
- Distribución geográfica
- Análisis de similitud
- Renta vs Área

💾 **Descargar:**
- Top 20 en formato CSV

---

## 🔍 Ejemplo Práctico

**Quiero abrir una tienda:**
- Segmento: "TRADICIONAL"
- Zona: "Norte"
- Municipio: "Bogotá"
- Estrato: 4
- Tipo: "LOCAL"
- Área: 150 m²
- Generador: "GENERADOR"

**La app me dice:**
> 🏆 Mejor Tienda Espejo: **"Unicentro"** (92% similitud)
> - VT: 2,500
> - ET: 5,200
> - Renta: $18,500

Ahora sé que mi tienda podría tener métricas similares a Unicentro.

---

## ❓ Preguntas Frecuentes

**P: ¿Es gratis Streamlit Cloud?**
R: Sí, 100% gratis para apps públicas.

**P: ¿Cuántas personas pueden usar la app?**
R: Ilimitadas. Puedes compartir la URL con todo tu equipo.

**P: ¿Puedo actualizar los datos?**
R: Sí, solo sube un nuevo Book.xlsx a GitHub y se actualiza automáticamente.

**P: ¿Necesito saber programar?**
R: No, solo seguir los pasos para subirlo a GitHub.

**P: ¿Qué pasa si no encuentro tiendas espejo?**
R: Verifica que existan tiendas en ese segmento en tu base de datos.

**P: ¿Puedo cambiar el algoritmo?**
R: Sí, modificando app.py y volviendo a subir a GitHub.

**P: ¿Se pueden agregar más columnas?**
R: Sí, pero necesitarías modificar el código en app.py.

---

## 🆘 Solución de Problemas

| Problema | Solución |
|----------|----------|
| No carga la app | Espera 3-5 minutos, refresca la página |
| Error "No module named..." | Verifica que requirements.txt esté en GitHub |
| No encuentra Book.xlsx | Asegúrate de que el archivo esté en el repositorio |
| Similitud siempre 0% | Verifica que el segmento exista en los datos |
| App muy lenta | Reduce el tamaño del archivo Excel o filtra datos |

---

## 📞 Soporte

1. Revisa el README.md completo
2. Verifica los logs en Streamlit Cloud
3. Consulta: https://docs.streamlit.io

---

## 🎯 Tips Pro

1. **Comparte la URL** con tu equipo
2. **Guarda los CSV** de cada análisis
3. **Compara múltiples propuestas** para decidir mejor
4. **Ajusta los pesos** según tu estrategia
5. **Revisa el Top 10**, no solo la primera opción

---

¡Éxito con tu herramienta! 🚀
