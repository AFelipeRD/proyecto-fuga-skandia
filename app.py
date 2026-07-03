from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Inicializamos la API
app = FastAPI(title="API de Prediccion de Fuga - Skandia", version="1.0")

# Cargamos el modelo y el preprocesador que creamos con entrenar.py
try:
    model = joblib.load('modelos/random_forest_model.pkl')
    preprocessor = joblib.load('modelos/preprocessor.pkl')
except Exception as e:
    print(f"Error al cargar los archivos del modelo: {e}")

# Estructura del JSON que el usuario debe enviarnos para predecir
class ClienteInput(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/health")
def estado_servicio():
    """Para revisar de forma sencilla que la API esté funcionando"""
    return {"status": "ok", "message": "API de predicciones activa y lista."}

@app.post("/predict")
def predecir_riesgo(cliente: ClienteInput):
    """Recibe datos de un cliente y calcula su probabilidad de fuga"""
    try:
        # Convertimos los datos que nos enviaron a un DataFrame
        datos_cliente = pd.DataFrame([cliente.model_dump()])
        
        # Agregamos los cálculos que creamos en el entrenamiento
        datos_cliente['balance_salary_ratio'] = datos_cliente['Balance'] / (datos_cliente['EstimatedSalary'] + 1)
        tasas_pais = {"France": 0.92, "Spain": 0.95, "Germany": 0.94}
        datos_cliente['tasa_retencion_pais'] = datos_cliente['Geography'].map(lambda x: tasas_pais.get(x, 0.90))
        
        # Aplicamos el preprocesamiento
        datos_proc = preprocessor.transform(datos_cliente)
        
        # Obtenemos la probabilidad y definimos la clasificación
        probabilidad = float(model.predict_proba(datos_proc)[0][1])
        clase_riesgo = "Alto Riesgo" if probabilidad >= 0.50 else "Bajo Riesgo"
        
        return {
            "probabilidad_fuga": round(probabilidad, 4),
            "clase_riesgo": clase_riesgo,
            "umbral_usado": 0.50,
            "explicacion": f"El cliente tiene un {round(probabilidad * 100, 2)}% de probabilidad de abandonar el servicio."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")