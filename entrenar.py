import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# 1. Cargamos el CSV y el archivo JSON (Cruzando fuentes de datos)
print("Cargando base de datos y archivo JSON de contexto...")
df = pd.read_csv('datos/Churn_Modelling.csv')

with open('datos/contexto_negocio.json', 'r') as f:
    contexto_json = json.load(f)

# Asociamos la tasa de retención del JSON usando el país del cliente
df['tasa_retencion_pais'] = df['Geography'].map(lambda x: contexto_json.get(x, {}).get('tasa_retencion_objetivo', 0.90))

# 2. Limpieza de datos básica
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname']) # Quitamos datos que no sirven para predecir (IDs, Apellidos)

# Feature Engineering (Creamos una variable sencilla)
df['balance_salary_ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)

# Clasificamos nuestras columnas
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'balance_salary_ratio', 'tasa_retencion_pais']
cat_cols = ['Geography', 'Gender']

X = df.drop(columns=['Exited'])
y = df['Exited']

# Separamos en entrenamiento y prueba (80% para aprender, 20% para probar al final)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Preprocesamiento (Escalar números y codificar categorías)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

X_train_proc = preprocessor.fit_transform(X_train)

# 4. Entrenamiento del modelo seleccionado (Random Forest)
print("Entrenando el modelo de Random Forest...")
model_rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
model_rf.fit(X_train_proc, y_train)

# 5. Guardamos el modelo entrenado y el transformador en la carpeta 'modelos'
print("Guardando archivos del modelo en la carpeta 'modelos'...")
joblib.dump(model_rf, 'modelos/random_forest_model.pkl')
joblib.dump(preprocessor, 'modelos/preprocessor.pkl')

print("¡Proceso de entrenamiento completado exitosamente!")