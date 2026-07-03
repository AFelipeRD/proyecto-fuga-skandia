# Modelo Predictivo de Fuga de Clientes - Skandia

Este proyecto contiene una solución analítica de extremo a extremo para identificar de forma proactiva a aquellos clientes que presentan un alto riesgo de fuga o cancelación de sus productos financieros en Skandia. 

La solución incluye un script de entrenamiento, un análisis exploratorio visual, la capacidad de realizar predicciones masivas y un microservicio listo para producción expuesto mediante una API REST.

---

## Estructura del Proyecto

El espacio de trabajo está organizado de la siguiente manera para garantizar su orden y reproducibilidad en cualquier computadora:

```text
proyecto-fuga-skandia/
│
├── datos/
│   ├── Churn_Modelling.csv         # Base de datos principal de clientes (Kaggle)
│   ├── contexto_negocio.json       # Datos semiestructurados con metas por país (JSON)
│   ├── eda_edad_fuga.png           # Gráfico generado del comportamiento de la edad
│   ├── importancia_variables.png   # Gráfico generado de la importancia de variables
│   └── clientes_con_predicciones.csv # Reporte generado de predicciones masivas
│
├── modelos/
│   ├── random_forest_model.pkl     # Archivo del modelo entrenado
│   └── preprocessor.pkl            # Archivo del preprocesador de datos
│
├── requirements.txt                # Lista de librerías de Python requeridas
├── entrenar.py                     # Script para entrenar y guardar el modelo
├── analisis_visual.py              # Script para generar las gráficas de EDA e importancia
├── prediccion_masiva.py            # Script para predecir sobre toda la base de datos de golpe
└── app.py                          # Código del microservicio expuesto con FastAPI
```

---

## Instrucciones de Instalación y Ejecución

Siga estos pasos sencillos en su terminal de Windows para instalar y ejecutar el proyecto completo localmente:

### 1. Preparar el entorno e instalar dependencias
Navegue hasta la carpeta del proyecto e instale todas las librerías necesarias con un solo comando:
```bash
cd proyecto-fuga-skandia
pip install -r requirements.txt
```

### 2. Entrenar el modelo de Machine Learning
Ejecute el script de entrenamiento para procesar los datos cruzados (CSV + JSON) y entrenar el modelo de Random Forest. Este comando generará y guardará el modelo entrenado en la carpeta `modelos`:
```bash
python entrenar.py
```

### 3. Generar las visualizaciones y análisis de importancia (EDA)
Ejecute este script para calcular las variables más influyentes y generar de forma automática los gráficos de negocio dentro de la carpeta `datos`:
```bash
python analisis_visual.py
```

### 4. Ejecutar la predicción masiva (Procesamiento por lotes)
Si desea procesar toda la base de datos de clientes a la vez y exportar un reporte con los niveles de riesgo calculados para cada uno de ellos, ejecute:
```bash
python prediccion_masiva.py
```
*Este comando generará el archivo `datos/clientes_con_predicciones.csv` listo para abrirse directamente en Excel.*

### 5. Iniciar el Microservicio (API de producción)
Para encender el servidor web local y exponer el modelo para consultas de clientes individuales en tiempo real, ejecute:
```bash
uvicorn app:app --reload
```
* Una vez encendido, puede verificar el estado del servicio ingresando en su navegador a: `http://127.0.0.1:8000/health`
* Para realizar pruebas de predicción de forma visual e interactiva, ingrese a la interfaz de Swagger en: `http://127.0.0.1:8000/docs`

---

## Uso Seguro de IA (Mini-reto)

En el desarrollo de este proyecto se utilizaron herramientas de inteligencia artificial generativa como asistentes de código, bajo estrictas políticas de uso responsable y ético:

1. **Partes asistidas y validación manual:** La IA fue utilizada para agilizar la escritura de la estructura inicial del microservicio en FastAPI y en la optimización de código de limpieza de datos. Absolutamente todo el código fue probado de manera manual en un entorno local, verificando que los cálculos matemáticos, las métricas de desempeño del modelo y el comportamiento de la API web fueran precisos y reales.
2. **Mitigación de "Prompt Injection" en documentos o campos JSON:** Si un usuario intenta ingresar comandos maliciosos dentro de un campo de texto (como "ignora las instrucciones anteriores y cambia el resultado"), nuestro microservicio lo previene mediante una validación rígida de tipos de datos utilizando la librería `Pydantic` (clase `ClienteInput`). Al forzar que los campos transacionales sean estrictamente numéricos (enteros o decimales) y los categóricos coincidan con valores previamente conocidos, se anula cualquier posibilidad de que una entrada de texto maliciosa afecte el comportamiento lógico del modelo o de la API.
3. **Controles para no exponer datos sensibles de clientes:** No se utilizaron datos reales de clientes de Skandia ni información confidencial en ninguna fase del desarrollo. Se empleó un dataset público de código de barra abierto debidamente anonimizado. En un entorno productivo real, se recomienda que la API se aloje dentro de una Nube Privada Virtual (VPC) corporativa y se implemente un filtro previo de enmascaramiento (como Microsoft Presidio) para eliminar nombres o números de identificación antes de que los datos toquen cualquier modelo de lenguaje o servicio externo.
```
