# Predicción de Precio de Casas

## Requisitos

Se necesita Python 3.12 o una versión posterior. El repositorio debe conservar las carpetas data y models, ya que contienen el dataset de entrenamiento, los preprocesadores y los pesos de las redes finales.

Para crear un entorno virtual e instalar las dependencias utilizadas:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows, la activación del entorno se realiza con:

```powershell
.venv\Scripts\activate
```

## Generar predicciones

El archivo de entrada debe ser un CSV con las mismas 80 columnas predictoras utilizadas durante el entrenamiento.

Para generar predicciones con el modelo individual de control:

```bash
python predecir.py --modelo control --entrada pipeline_test.csv --salida predicciones_control.csv
```

Para utilizar el ensemble de cinco redes:

```bash
python predecir.py --modelo ensemble --entrada pipeline_test.csv --salida predicciones_ensemble.csv
```
