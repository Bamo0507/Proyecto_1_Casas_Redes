import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn


RAIZ = Path(__file__).resolve().parent
DIRECTORIO_MODELOS = RAIZ / "models" / "pytorch" / "evaluacion_final_test"
RUTA_PREPROCESADOR = DIRECTORIO_MODELOS / "preprocesador.joblib"
DISPOSITIVO = torch.device("cpu")

COLUMNAS_ENTRADA = [
    "Id",
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "MasVnrArea",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinSF1",
    "BsmtFinType2",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Functional",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MiscVal",
    "MoSold",
    "YrSold",
    "SaleType",
    "SaleCondition",
]

RELLENOS_ESTRUCTURALES = {
    "Alley": "SinCallejon",
    "BsmtQual": "SinSotano",
    "BsmtCond": "SinSotano",
    "BsmtExposure": "SinSotano",
    "BsmtFinType1": "SinSotano",
    "BsmtFinType2": "SinSotano",
    "FireplaceQu": "SinChimenea",
    "GarageType": "SinGaraje",
    "GarageFinish": "SinGaraje",
    "GarageQual": "SinGaraje",
    "GarageCond": "SinGaraje",
    "PoolQC": "SinPiscina",
    "Fence": "SinCerca",
    "MiscFeature": "SinCaracteristica",
}


class TransformadorAmes(BaseEstimator, TransformerMixin):
    def __init__(self, eliminar_anios_originales=False):
        self.eliminar_anios_originales = eliminar_anios_originales

    def fit(self, X, y=None):
        datos = X.copy()
        self.mediana_frente_vecindario_ = datos.groupby("Neighborhood")[
            "LotFrontage"
        ].median()
        self.mediana_frente_global_ = datos["LotFrontage"].median()
        moda_electrical = datos["Electrical"].mode(dropna=True)
        self.moda_electrical_ = (
            moda_electrical.iloc[0]
            if not moda_electrical.empty
            else "Desconocido"
        )
        return self

    def transform(self, X):
        datos = X.copy()
        columnas_texto = datos.select_dtypes(include="object").columns
        for columna in columnas_texto:
            datos[columna] = (
                datos[columna]
                .astype("string")
                .str.strip()
                .str.strip("'")
            )

        for columna, etiqueta in RELLENOS_ESTRUCTURALES.items():
            datos[columna] = datos[columna].fillna(etiqueta)

        datos["LotFrontageMissing"] = datos["LotFrontage"].isna().astype(int)
        medianas_fila = datos["Neighborhood"].map(
            self.mediana_frente_vecindario_
        )
        datos["LotFrontage"] = (
            datos["LotFrontage"]
            .fillna(medianas_fila)
            .fillna(self.mediana_frente_global_)
        )
        datos["MasVnrType"] = datos["MasVnrType"].fillna("Desconocido")
        datos["MasVnrArea"] = datos["MasVnrArea"].fillna(0)
        datos["Electrical"] = datos["Electrical"].fillna(
            self.moda_electrical_
        )
        datos["HasGarage"] = datos["GarageCars"].gt(0).astype(int)
        datos["GarageAge"] = datos["YrSold"] - datos["GarageYrBlt"]
        datos.loc[datos["HasGarage"].eq(0), "GarageAge"] = 0
        datos["GarageAge"] = datos["GarageAge"].fillna(0)
        datos["HouseAge"] = (
            datos["YrSold"] - datos["YearBuilt"]
        ).clip(lower=0)
        edad_remodelacion = datos["YrSold"] - datos["YearRemodAdd"]
        datos["RemodAfterSale"] = edad_remodelacion.lt(0).astype(int)
        datos["RemodAge"] = edad_remodelacion.clip(lower=0)
        datos["MSSubClass"] = datos["MSSubClass"].astype("Int64").astype("string")

        columnas_restantes_texto = datos.select_dtypes(
            include=["object", "string"]
        ).columns
        for columna in columnas_restantes_texto:
            datos[columna] = datos[columna].fillna("Desconocido")

        columnas_eliminar = ["Id", "GarageYrBlt"]
        if self.eliminar_anios_originales:
            columnas_eliminar.extend(["YearBuilt", "YearRemodAdd", "YrSold"])
        return datos.drop(columns=columnas_eliminar)


class TransformadorLogSesgo(BaseEstimator, TransformerMixin):
    def __init__(self, umbral=1.0, habilitado=True):
        self.umbral = umbral
        self.habilitado = habilitado

    def fit(self, X, y=None):
        datos = pd.DataFrame(X).copy()
        self.columnas_transformadas_ = []
        if self.habilitado:
            for columna in datos.columns:
                serie = pd.to_numeric(datos[columna], errors="coerce")
                valores_observados = serie.dropna()
                if (
                    not valores_observados.empty
                    and valores_observados.min() >= 0
                    and valores_observados.nunique() > 10
                    and abs(valores_observados.skew()) > self.umbral
                ):
                    self.columnas_transformadas_.append(columna)
        return self

    def transform(self, X):
        datos = pd.DataFrame(X).copy()
        for columna in self.columnas_transformadas_:
            datos[columna] = np.log1p(datos[columna])
        return datos

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def seleccionar_categoricas(datos):
    return datos.select_dtypes(exclude=np.number).columns.tolist()


class MLPPrecios(nn.Module):
    def __init__(
        self,
        dimension_entrada,
        capas_ocultas,
        activacion="relu",
        dropout=0.0,
        batch_norm=False,
        inicializacion="original",
        pendiente_leaky=0.01,
    ):
        super().__init__()
        activaciones = {
            "relu": lambda: nn.ReLU(),
            "leaky_relu": lambda: nn.LeakyReLU(pendiente_leaky),
            "tanh": lambda: nn.Tanh(),
        }
        capas = []
        dimension_anterior = dimension_entrada
        for neuronas in capas_ocultas:
            capa_lineal = nn.Linear(dimension_anterior, neuronas)
            capas.append(capa_lineal)
            if batch_norm:
                capas.append(nn.BatchNorm1d(neuronas))
            capas.append(activaciones[activacion]())
            if dropout > 0:
                capas.append(nn.Dropout(dropout))
            dimension_anterior = neuronas
        capas.append(nn.Linear(dimension_anterior, 1))
        self.red = nn.Sequential(*capas)

    def forward(self, X):
        return self.red(X).squeeze(-1)


def validar_entrada(datos):
    if datos.empty:
        raise ValueError("El archivo de entrada no contiene observaciones.")
    faltantes = [
        columna for columna in COLUMNAS_ENTRADA if columna not in datos.columns
    ]
    adicionales = [
        columna for columna in datos.columns if columna not in COLUMNAS_ENTRADA
    ]
    if faltantes or adicionales:
        raise ValueError(
            "El esquema de entrada no coincide. "
            f"Faltantes: {faltantes}. Adicionales: {adicionales}."
        )
    if datos["Id"].isna().any():
        raise ValueError("La columna Id contiene valores faltantes.")
    if datos["Id"].duplicated().any():
        raise ValueError("La columna Id contiene valores duplicados.")


def cargar_modelo(ruta_modelo):
    paquete = torch.load(
        ruta_modelo,
        map_location=DISPOSITIVO,
        weights_only=False,
    )
    configuracion = paquete["configuracion"]
    modelo = MLPPrecios(
        dimension_entrada=paquete["dimension_entrada"],
        capas_ocultas=configuracion["capas_ocultas"],
        activacion=configuracion["activacion"],
        dropout=configuracion["dropout"],
        batch_norm=configuracion["batch_norm"],
        inicializacion=configuracion.get("inicializacion", "original"),
        pendiente_leaky=configuracion.get("pendiente_leaky", 0.01),
    ).to(DISPOSITIVO)
    modelo.load_state_dict(paquete["state_dict"])
    modelo.eval()
    return modelo


def predecir_modelo(modelo, X):
    with torch.inference_mode():
        tensor_X = torch.as_tensor(X, dtype=torch.float32, device=DISPOSITIVO)
        prediccion_transformada = modelo(tensor_X).cpu().numpy()
    return np.clip(np.expm1(prediccion_transformada), a_min=0, a_max=None)


def generar_predicciones(ruta_entrada, ruta_salida, nombre_modelo):
    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta_entrada}")
    if not RUTA_PREPROCESADOR.exists():
        raise FileNotFoundError(f"No existe el preprocesador: {RUTA_PREPROCESADOR}")

    datos = pd.read_csv(
        ruta_entrada,
        keep_default_na=False,
        na_values=["", "NA"],
    )
    print(f"[cargado]  {ruta_entrada} -> {len(datos)} filas, {datos.shape[1]} columnas")
    validar_entrada(datos)
    datos = datos[COLUMNAS_ENTRADA].copy()
    print("[ok]       El esquema de entrada es valido")

    preprocesador = joblib.load(RUTA_PREPROCESADOR)
    X_procesado = np.asarray(
        preprocesador.transform(datos),
        dtype=np.float32,
    )

    cantidad_miembros = 1 if nombre_modelo == "control" else 5
    predicciones = []
    for numero in range(1, cantidad_miembros + 1):
        ruta_modelo = DIRECTORIO_MODELOS / f"miembro_{numero}_modelo.pt"
        if not ruta_modelo.exists():
            raise FileNotFoundError(f"No existe el modelo: {ruta_modelo}")
        modelo = cargar_modelo(ruta_modelo)
        predicciones.append(predecir_modelo(modelo, X_procesado))

    prediccion_final = np.mean(np.vstack(predicciones), axis=0)
    if not np.isfinite(prediccion_final).all():
        raise ValueError("Las predicciones contienen valores no finitos.")

    salida = pd.DataFrame(
        {
            "Id": datos["Id"].to_numpy(),
            "Prediction": prediccion_final,
        }
    )
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta_salida, index=False)
    print(f"[guardado] {ruta_salida} -> {len(salida)} filas, 2 columnas")


def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Genera predicciones con el modelo control o ensemble."
    )
    parser.add_argument(
        "--modelo",
        choices=["control", "ensemble"],
        required=True,
    )
    parser.add_argument("--entrada", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    return parser.parse_args()


def main():
    argumentos = obtener_argumentos()
    generar_predicciones(
        ruta_entrada=argumentos.entrada,
        ruta_salida=argumentos.salida,
        nombre_modelo=argumentos.modelo,
    )


main()
