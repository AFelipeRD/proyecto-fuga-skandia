import pandas as pd
import joblib
import json

print("Iniciando proceso de predicción masiva...")

# 1. Cargamos el modelo, el preprocesador y el archivo de clientes nuevos
model = joblib.load('modelos/random_forest_model.pkl')
preprocessor = joblib.load('modelos/preprocessor.pkl')
df_nuevos = pd.read_csv('datos/Churn_Modelling.csv') # Aquí cargamos la tabla completa

# 2. Hacemos el Feature Engineering automático (Cruce con el JSON y ratio de balance)
df_nuevos['balance_salary_ratio'] = df_nuevos['Balance'] / (df_nuevos['EstimatedSalary'] + 1)

with open('datos/contexto_negocio.json', 'r') as f:
    contexto = json.load(f)
df_nuevos['tasa_retencion_pais'] = df_nuevos['Geography'].map(lambda x: contexto.get(x, {}).get('tasa_retencion_objetivo', 0.90))

# 3. Limpiamos los datos del ID y datos no predictivos
X_datos = df_nuevos.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'], errors='ignore')

# 4. Aplicamos el preprocesamiento a toda la tabla a la vez
X_proc = preprocessor.transform(X_datos)

# 5. Calculamos la probabilidad de fuga para TODOS los clientes en un segundo
df_nuevos['Probabilidad_Fuga'] = model.predict_proba(X_proc)[:, 1]
df_nuevos['Alerta_Riesgo'] = df_nuevos['Probabilidad_Fuga'].map(lambda p: 'Alto Riesgo' if p >= 0.50 else 'Bajo Riesgo')

# 6. Guardamos los resultados en un nuevo archivo CSV
df_nuevos.to_csv('datos/clientes_con_predicciones.csv', index=False)

print("¡Listo! Se ha generado el archivo 'datos/clientes_con_predicciones.csv' con las alertas para todos los clientes.")