import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# ============================================================
# 1. ACTIVOS
# ============================================================

tickers = {
    "YPF": "YPF",
    "MERCADOLIBRE": "MELI",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "NVIDIA": "NVDA"
}

# ============================================================
# 2. OBTENER PRECIOS HISTÓRICOS
# ============================================================

def obtener_precios(tickers, period="5y", interval="1mo"):
    """
    Obtiene los precios históricos de los activos.
    """

    prices = yf.download(
        list(tickers.values()),
        period=period,
        interval=interval,
        auto_adjust=True
    )["Close"]

    # Reemplazamos los tickers de Yahoo por los nombres
    # que utilizamos en nuestro modelo
    prices.columns = tickers.keys()

    return prices


def calcular_rendimientos(precios):
    """
    Calcula los rendimientos simples mensuales
    a partir de los precios históricos.
    """

    return precios.pct_change().dropna()

# ============================================================
# 3. PARÁMETROS DEL MODELO
# ============================================================

TAM_CARTERA = 5
TAM_POBLACION = 100
GENERACIONES = 1000

PROB_MUTACION = 0.05
DESVIACION_MUTACION = 0.05

PROB_CROSSOVER = 0.8

LAMBDA = 1 # Coeficiente de aversión al riesgo

# Tasa libre de riesgo, necesaria si usamos Sharpe
RISK_FREE_RATE = 0.0


# ============================================================
# 5. GENERAR UNA CARTERA
# ============================================================

def generar_cartera():
    """
    Genera un individuo.

    Cada gen representa el porcentaje invertido
    en un activo.
    """

    proporciones = np.random.random(TAM_CARTERA)

    # Normalizamos para que la suma sea 1
    proporciones /= np.sum(proporciones)

    return proporciones


# ============================================================
# 6. GENERAR POBLACIÓN
# ============================================================

def generar_poblacion():

    return np.array([
        generar_cartera()
        for _ in range(TAM_POBLACION)
    ])


# ============================================================
# 7. RENDIMIENTO DE LA CARTERA
# ============================================================

def retorno_cartera(proporciones):

    return np.dot(proporciones, mu)


# ============================================================
# 8. RIESGO DE LA CARTERA
# ============================================================

def riesgo_cartera(proporciones):

    varianza = proporciones.T @ cov_matrix @ proporciones

    return np.sqrt(varianza)


# ============================================================
# 9. FITNESS
# ============================================================

def fitness(proporciones):

    expected_return = retorno_cartera(proporciones)

    riesgo = riesgo_cartera(proporciones)

    return expected_return - LAMBDA * riesgo


# ============================================================
# 10. SELECCIÓN
# ============================================================

def seleccion_truncamiento(poblacion):

    fitness_values = np.array([
        fitness(individual)
        for individual in poblacion
    ])

    # Ordenamos de mejor a peor
    indexes = np.argsort(fitness_values)[::-1]

    # Seleccionamos la mitad superior
    selected = poblacion[
        indexes[:TAM_POBLACION // 2]
    ]

    return selected


# ============================================================
# 11. CRUCE
# ============================================================

def crossover(p1, p2):

    if np.random.random() > PROB_CROSSOVER:

        return p1.copy()

    alpha = np.random.random()

    child = (
        alpha * p1
        + (1 - alpha) * p2
    )

    # Nos aseguramos de que los pesos sumen 1
    child /= np.sum(child)

    return child


# ============================================================
# 12. MUTACIÓN
# ============================================================

def mutacion(individuo):

    for i in range(TAM_CARTERA):

        if np.random.random() < PROB_MUTACION:

            # Pequeña modificación del peso
            individuo[i] += np.random.normal(0, DESVIACION_MUTACION)

    # Evitamos pesos negativos
    individuo = np.maximum(individuo, 0)

    # Volvemos a normalizar
    if np.sum(individuo) > 0:

        individuo /= np.sum(individuo)

    return individuo


# ============================================================
# 13. CREAR NUEVA POBLACIÓN
# ============================================================

def crear_nueva_poblacion(poblacion_selec):

    nueva_poblacion = []

    while len(nueva_poblacion) < TAM_POBLACION:

        parent1 = poblacion_selec[
            np.random.randint(len(poblacion_selec))
        ]

        parent2 = poblacion_selec[
            np.random.randint(len(poblacion_selec))
        ]

        child = crossover(
            parent1,
            parent2
        )

        child = mutacion(child)

        nueva_poblacion.append(child)

    return np.array(nueva_poblacion)


# ============================================================
# 14. ALGORITMO GENÉTICO
# ============================================================

def algoritmo_genetico():

    poblacion = generar_poblacion()

    mejor_cromosoma = None
    mejor_fitness = -np.inf

    for generacion in range(GENERACIONES):

        # Evaluamos la población
        valores_fitness = np.array([
            fitness(individual)
            for individual in poblacion
        ])

        # Mejor individuo de esta generación
        mejor_indice = np.argmax(valores_fitness)

        mejor_actual = poblacion[mejor_indice]

        fitness_actual = valores_fitness[mejor_indice]

        # Guardamos el mejor global
        if fitness_actual > mejor_fitness:

            mejor_fitness = fitness_actual

            mejor_cromosoma = mejor_actual.copy()

        # TODO: Implementar elitismo 
        # TODO: Implementar mas estrategias de selección
        
        selected = seleccion_truncamiento(poblacion)

        # Cruce + mutación
        poblacion = crear_nueva_poblacion(selected)

        print(
            f"Generación {generacion + 1}: "
            f"Fitness = {fitness_actual:.6f}"
        )

    return mejor_cromosoma, mejor_fitness


def graficar_cartera(mejor_cartera):

    plt.figure(figsize=(8, 8))

    plt.pie(
        mejor_cartera,
        labels=retornos.columns,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Distribución de la cartera óptima")
    plt.show()

# ============================================================
# 15. EJECUCIÓN
# ============================================================

# Obtenemos los precios
precios = obtener_precios(tickers)

# Calculamos los rendimientos
retornos = calcular_rendimientos(precios)

# Rendimiento esperado de cada activo
mu = retornos.mean().values

# Matriz de covarianzas
cov_matrix = retornos.cov().values

mejor_cartera, mejor_fitness = algoritmo_genetico()

# ============================================================
# 16. RESULTADOS
# ============================================================

print("\n--- MEJOR CARTERA ---")

for asset, weight in zip(
    retornos.columns,
    mejor_cartera
):

    print(
        f"{asset}: {weight:.2%}"
    )


print(
    "\nRendimiento esperado:",
    retorno_cartera(mejor_cartera)
)

print(
    "Riesgo:",
    riesgo_cartera(mejor_cartera)
)

print(
    "Fitness:",
    mejor_fitness
)


print("\n--- DATOS OBTENIDOS ---")
print(retornos)

# Resultados mensuales
rendimiento_mensual = retorno_cartera(mejor_cartera)
riesgo_mensual = riesgo_cartera(mejor_cartera)

# Anualización
rendimiento_anual = (1 + rendimiento_mensual)**12 - 1
riesgo_anual = riesgo_mensual * np.sqrt(12)

graficar_cartera(mejor_cartera)

print("Rendimiento esperado mensual:", np.round(rendimiento_mensual * 100, 2))
print("Riesgo mensual:", np.round(riesgo_mensual * 100, 2))

print("Rendimiento esperado anual:", np.round(rendimiento_anual * 100, 2))
print("Riesgo anual:", np.round(riesgo_anual * 100, 2))