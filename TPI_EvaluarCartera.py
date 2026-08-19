import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# 1. ACTIVOS
# ============================================================

activos = {
    "YPF": "YPF",
    "MICROSOFT": "MSFT",
    "APPLE": "AAPL",
    "MERCADOLIBRE": "MELI",
    "NVIDIA": "NVDA"
}


# ============================================================
# 2. OBTENER PRECIOS HISTÓRICOS
# ============================================================

def obtener_precios(activos, periodo="5y", intervalo="1mo"):
    """
    Obtiene los precios históricos de los activos.
    """

    precios = yf.download(
        list(activos.values()),
        period=periodo,
        interval=intervalo,
        auto_adjust=True
    )["Close"]

    # Reemplazamos los identificadores de Yahoo
    # por los nombres utilizados en nuestro modelo
    precios.columns = activos.keys()

    return precios


def calcular_rendimientos(precios):
    """
    Calcula los rendimientos simples mensuales
    a partir de los precios históricos.
    """

    return precios.pct_change().dropna()

# ============================================================
# 3. PARÁMETROS DEL MODELO
# ============================================================

TAMANO_CARTERA = 5
TAMANO_POBLACION = 100
CANTIDAD_GENERACIONES = 500

PROBABILIDAD_MUTACION = 0.05
DESVIO_ESTANDAR_MUTACION = 0.05

PROBABILIDAD_CRUCE = 0.8

LAMBDA = 10

# Tasa libre de riesgo, necesaria si utilizamos
# el índice de Sharpe
TASA_LIBRE_RIESGO = 0.0


# ============================================================
# 5. GENERAR UNA CARTERA
# ============================================================

def generar_cartera():
    """
    Genera un individuo.

    Cada gen representa la proporción del capital
    invertida en un activo.
    """

    proporciones = np.random.random(TAMANO_CARTERA)

    # Normalizamos para que la suma sea 1
    proporciones /= np.sum(proporciones)

    return proporciones


# ============================================================
# 6. GENERAR POBLACIÓN
# ============================================================

def generar_poblacion():

    return np.array([
        generar_cartera()
        for _ in range(TAMANO_POBLACION)
    ])


# ============================================================
# 7. RENDIMIENTO DE LA CARTERA
# ============================================================

def calcular_rendimiento_cartera(proporciones):

    return np.dot(
        proporciones,
        rendimiento_esperado
    )


# ============================================================
# 8. RIESGO DE LA CARTERA
# ============================================================

def calcular_riesgo_cartera(proporciones):

    varianza = (
        proporciones.T
        @ matriz_covarianzas
        @ proporciones
    )

    return np.sqrt(varianza)


# ============================================================
# 9. EVALUACIÓN DE LA CARTERA
# ============================================================

def evaluar_cartera(proporciones):

    rendimiento = calcular_rendimiento_cartera(
        proporciones
    )

    riesgo = calcular_riesgo_cartera(
        proporciones
    )

    return rendimiento - LAMBDA * riesgo


# ============================================================
# 10. SELECCIÓN
# ============================================================

def seleccionar(poblacion):

    valores_evaluacion = np.array([
        evaluar_cartera(individuo)
        for individuo in poblacion
    ])

    # Ordenamos de mejor a peor
    indices = np.argsort(
        valores_evaluacion
    )[::-1]

    # Seleccionamos la mitad superior
    seleccionados = poblacion[
        indices[:TAMANO_POBLACION // 2]
    ]

    return seleccionados


# ============================================================
# 11. CRUCE
# ============================================================

def cruzar(progenitor1, progenitor2):

    if np.random.random() > PROBABILIDAD_CRUCE:

        return progenitor1.copy()

    proporcion = np.random.random()

    descendiente = (
        proporcion * progenitor1
        + (1 - proporcion) * progenitor2
    )

    # Nos aseguramos de que las proporciones sumen 1
    descendiente /= np.sum(descendiente)

    return descendiente


# ============================================================
# 12. MUTACIÓN
# ============================================================

def mutar(individuo):

    for i in range(TAMANO_CARTERA):

        if np.random.random() < PROBABILIDAD_MUTACION:

            # Pequeña modificación de la proporción
            individuo[i] += np.random.normal(
                0,
                DESVIO_ESTANDAR_MUTACION
            )

    # Evitamos proporciones negativas
    individuo = np.maximum(individuo, 0)

    # Volvemos a normalizar
    if np.sum(individuo) > 0:

        individuo /= np.sum(individuo)

    return individuo


# ============================================================
# 13. CREAR NUEVA POBLACIÓN
# ============================================================

def crear_nueva_poblacion(seleccionados):

    nueva_poblacion = []

    while len(nueva_poblacion) < TAMANO_POBLACION:

        progenitor1 = seleccionados[
            np.random.randint(
                len(seleccionados)
            )
        ]

        progenitor2 = seleccionados[
            np.random.randint(
                len(seleccionados)
            )
        ]

        descendiente = cruzar(
            progenitor1,
            progenitor2
        )

        descendiente = mutar(
            descendiente
        )

        nueva_poblacion.append(
            descendiente
        )

    return np.array(nueva_poblacion)


# ============================================================
# 14. ALGORITMO GENÉTICO
# ============================================================

def ejecutar_algoritmo_genetico():

    poblacion = generar_poblacion()

    mejor_cartera = None
    mejor_evaluacion = -np.inf

    for generacion in range(
        CANTIDAD_GENERACIONES
    ):

        # Evaluamos la población
        valores_evaluacion = np.array([
            evaluar_cartera(individuo)
            for individuo in poblacion
        ])

        # Mejor individuo de esta generación
        mejor_indice = np.argmax(
            valores_evaluacion
        )

        mejor_actual = poblacion[
            mejor_indice
        ]

        evaluacion_actual = valores_evaluacion[
            mejor_indice
        ]

        # Guardamos la mejor cartera encontrada
        if evaluacion_actual > mejor_evaluacion:

            mejor_evaluacion = evaluacion_actual

            mejor_cartera = mejor_actual.copy()

        # Selección
        seleccionados = seleccionar(
            poblacion
        )

        # Cruce y mutación
        poblacion = crear_nueva_poblacion(
            seleccionados
        )

        print(
            f"Generación {generacion + 1}: "
            f"Evaluación = {evaluacion_actual:.6f}"
        )

    return mejor_cartera, mejor_evaluacion


# ============================================================
# 15. EJECUCIÓN
# ============================================================

# Obtenemos los precios
precios = obtener_precios(activos)

# Calculamos los rendimientos
rendimientos = calcular_rendimientos(precios)

# Rendimiento esperado de cada activo
rendimiento_esperado = rendimientos.mean().values

# Matriz de covarianzas
matriz_covarianzas = rendimientos.cov().values

mejor_cartera, mejor_evaluacion = (
    ejecutar_algoritmo_genetico()
)


# ============================================================
# 16. RESULTADOS
# ============================================================

print("\n--- MEJOR CARTERA ---")

for activo, proporcion in zip(
    rendimientos.columns,
    mejor_cartera
):
    print(
        f"{activo}: {proporcion * 100:.2f}%"
    )

print(
    f"\nRendimiento esperado: "
    f"{calcular_rendimiento_cartera(mejor_cartera) * 100:.2f}%"
)

print(
    f"Riesgo: "
    f"{calcular_riesgo_cartera(mejor_cartera) * 100:.2f}%"
)

print(
    f"Evaluación: "
    f"{mejor_evaluacion:.6f}"
)