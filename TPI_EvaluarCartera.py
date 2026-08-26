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
    # utilizados en nuestro modelo.
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
GENERACIONES = 100

PROB_MUTACION = 0.05
DESVIACION_MUTACION = 0.05

PROB_CROSSOVER = 0.8

# Coeficiente de aversión al riesgo
LAMBDA = 0.5

# Tasa libre de riesgo mensual
# Se utiliza para el cálculo del Índice de Sharpe.
RISK_FREE_RATE = 0.0


# ============================================================
# 4. GENERAR UNA CARTERA
# ============================================================

def generar_cartera():
    """
    Genera un individuo.

    Cada gen representa el porcentaje invertido
    en un activo.
    """

    proporciones = np.random.random(TAM_CARTERA)

    # Normalizamos para que la suma sea 1.
    proporciones /= np.sum(proporciones)

    return proporciones


# ============================================================
# 5. GENERAR POBLACIÓN
# ============================================================

def generar_poblacion():

    return np.array([
        generar_cartera()
        for _ in range(TAM_POBLACION)
    ])


# ============================================================
# 6. RENDIMIENTO DE LA CARTERA
# ============================================================

def retorno_cartera(proporciones):

    return np.dot(proporciones, mu)


# ============================================================
# 7. RIESGO DE LA CARTERA
# ============================================================

def riesgo_cartera(proporciones):

    varianza = proporciones.T @ cov_matrix @ proporciones

    return np.sqrt(varianza)


# ============================================================
# 8. FITNESS - LAMBDA
# ============================================================

def fitness_lambda(proporciones):

    expected_return = retorno_cartera(proporciones)

    riesgo = riesgo_cartera(proporciones)

    return expected_return - LAMBDA * riesgo


# ============================================================
# 9. FITNESS - ÍNDICE DE SHARPE
# ============================================================

def fitness_sharpe(proporciones):

    expected_return = retorno_cartera(proporciones)

    riesgo = riesgo_cartera(proporciones)

    # Evitamos división por cero.
    if riesgo == 0:
        return 0

    return (
        expected_return - RISK_FREE_RATE
    ) / riesgo


# ============================================================
# 10. SELECCIÓN
# ============================================================

def seleccion_truncamiento(poblacion, fitness_function):

    fitness_values = np.array([
        fitness_function(individual)
        for individual in poblacion
    ])

    # Ordenamos de mejor a peor.
    indexes = np.argsort(fitness_values)[::-1]

    # Seleccionamos la mitad superior.
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

    hijo = (
        alpha * p1
        + (1 - alpha) * p2
    )

    # Nos aseguramos de que los pesos sumen 1.
    hijo /= np.sum(hijo)

    return hijo


# ============================================================
# 12. MUTACIÓN
# ============================================================

def mutacion(individuo):

    # Copiamos para no modificar directamente
    # al individuo de la población original.
    individuo = individuo.copy()

    for i in range(TAM_CARTERA):

        if np.random.random() < PROB_MUTACION:

            # Pequeña modificación del peso.
            individuo[i] += np.random.normal(
                0,
                DESVIACION_MUTACION
            )

    # Evitamos pesos negativos.
    individuo = np.maximum(individuo, 0)

    # Volvemos a normalizar.
    if np.sum(individuo) > 0:

        individuo /= np.sum(individuo)

    else:

        individuo = generar_cartera()

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

def algoritmo_genetico(
    fitness_function,
    poblacion_inicial=None
):
    """
    Ejecuta el algoritmo genético utilizando
    la función de fitness indicada.

    Si se proporciona una población inicial,
    se utiliza una copia de ella.
    """

    if poblacion_inicial is None:

        population = generar_poblacion()

    else:

        population = poblacion_inicial.copy()

    best_individual = None
    best_fitness = -np.inf

    for generation in range(GENERACIONES):

        # Evaluamos la población.
        fitness_values = np.array([
            fitness_function(individual)
            for individual in population
        ])

        # Mejor individuo de la generación.
        best_index = np.argmax(fitness_values)

        current_best = population[best_index]

        current_fitness = fitness_values[best_index]

        # Guardamos el mejor individuo global.
        if current_fitness > best_fitness:

            best_fitness = current_fitness

            best_individual = current_best.copy()

        # Selección.
        selected = seleccion_truncamiento(
            population,
            fitness_function
        )

        # Cruce + mutación.
        population = crear_nueva_poblacion(
            selected
        )

        print(
            f"Generación {generation + 1}: "
            f"Fitness = {current_fitness:.6f}"
        )

    return best_individual, best_fitness


# ============================================================
# 15. MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(
    titulo,
    portfolio,
    fitness_value
):

    expected_return = retorno_cartera(portfolio)

    risk = riesgo_cartera(portfolio)

    sharpe = fitness_sharpe(portfolio)

    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)

    print("\nDistribución de la cartera:")

    for asset, weight in zip(
        retornos.columns,
        portfolio
    ):

        print(
            f"{asset:<20}: {weight:>8.2%}"
        )

    print("\nRendimiento esperado:")
    print(f"{expected_return:.6f}")

    print("\nRiesgo:")
    print(f"{risk:.6f}")

    print("\nÍndice de Sharpe:")
    print(f"{sharpe:.6f}")

    print("\nFitness:")
    print(f"{fitness_value:.6f}")


# ============================================================
# 16. GRAFICAR CARTERA
# ============================================================

def graficar_cartera(
    mejor_cartera,
    titulo="Distribución de la cartera óptima"
):

    rendimiento_mensual = retorno_cartera(
        mejor_cartera
    )

    riesgo_mensual = riesgo_cartera(
        mejor_cartera
    )

    # Tasa anual simple.
    rendimiento_anual_tda = (
        rendimiento_mensual * 12
    )

    # Tasa efectiva anual.
    rendimiento_anual_tea = (
        (1 + rendimiento_mensual) ** 12 - 1
    )

    # Anualización de la volatilidad.
    riesgo_anual = (
        riesgo_mensual * np.sqrt(12)
    )

    plt.figure(figsize=(8, 8))

    plt.pie(
        mejor_cartera,
        labels=retornos.columns,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(titulo)

    descripcion = (
        f"Rendimiento esperado mensual: "
        f"{rendimiento_mensual * 100:.2f}%\n"
        f"Riesgo mensual: "
        f"{riesgo_mensual * 100:.2f}%\n\n"
        f"Rendimiento esperado anual (TDA): "
        f"{rendimiento_anual_tda * 100:.2f}%\n"
        f"Rendimiento esperado anual (TEA): "
        f"{rendimiento_anual_tea * 100:.2f}%\n"
        f"Riesgo anual: "
        f"{riesgo_anual * 100:.2f}%"
    )

    plt.figtext(
        0.5,
        0.02,
        descripcion,
        ha="center",
        fontsize=11
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 1]
    )

    plt.show()


# ============================================================
# 17. COMPARAR MÉTODOS
# ============================================================

def comparar_metodos():

    print("\n")
    print("=" * 60)
    print("              COMPARACIÓN DE MÉTODOS")
    print("=" * 60)

    # --------------------------------------------------------
    # Misma población inicial para ambos métodos.
    # --------------------------------------------------------

    poblacion_inicial = generar_poblacion()

    # --------------------------------------------------------
    # FITNESS LAMBDA
    # --------------------------------------------------------

    print("\nEjecutando Fitness Lambda...\n")

    portfolio_lambda, fitness_lambda_value = (
        algoritmo_genetico(
            fitness_lambda,
            poblacion_inicial
        )
    )

    # --------------------------------------------------------
    # FITNESS SHARPE
    # --------------------------------------------------------

    print("\nEjecutando Índice de Sharpe...\n")

    portfolio_sharpe, fitness_sharpe_value = (
        algoritmo_genetico(
            fitness_sharpe,
            poblacion_inicial
        )
    )

    # --------------------------------------------------------
    # RESULTADOS INDIVIDUALES
    # --------------------------------------------------------

    mostrar_resultados(
        "RESULTADO - FITNESS LAMBDA",
        portfolio_lambda,
        fitness_lambda_value
    )

    mostrar_resultados(
        "RESULTADO - ÍNDICE DE SHARPE",
        portfolio_sharpe,
        fitness_sharpe_value
    )

    # --------------------------------------------------------
    # MÉTRICAS COMPARABLES
    # --------------------------------------------------------

    return_lambda = retorno_cartera(
        portfolio_lambda
    )

    risk_lambda = riesgo_cartera(
        portfolio_lambda
    )

    sharpe_lambda = fitness_sharpe(
        portfolio_lambda
    )

    return_sharpe = retorno_cartera(
        portfolio_sharpe
    )

    risk_sharpe = riesgo_cartera(
        portfolio_sharpe
    )

    sharpe_sharpe = fitness_sharpe(
        portfolio_sharpe
    )

    # --------------------------------------------------------
    # TABLA COMPARATIVA
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                         COMPARACIÓN")
    print("=" * 70)

    print(
        f"{'Métrica':<30}"
        f"{'Lambda':>18}"
        f"{'Sharpe':>18}"
    )

    print("-" * 70)

    print(
        f"{'Rendimiento esperado':<30}"
        f"{return_lambda:>18.6f}"
        f"{return_sharpe:>18.6f}"
    )

    print(
        f"{'Riesgo':<30}"
        f"{risk_lambda:>18.6f}"
        f"{risk_sharpe:>18.6f}"
    )

    print(
        f"{'Índice de Sharpe':<30}"
        f"{sharpe_lambda:>18.6f}"
        f"{sharpe_sharpe:>18.6f}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # PESOS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                    DISTRIBUCIÓN DE CARTERAS")
    print("=" * 70)

    print(
        f"{'Activo':<20}"
        f"{'Lambda':>20}"
        f"{'Sharpe':>20}"
    )

    print("-" * 70)

    for asset, weight_lambda, weight_sharpe in zip(
        retornos.columns,
        portfolio_lambda,
        portfolio_sharpe
    ):

        print(
            f"{asset:<20}"
            f"{weight_lambda:>19.2%}"
            f"{weight_sharpe:>20.2%}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # GRÁFICOS
    # --------------------------------------------------------

    graficar_cartera(
        portfolio_lambda,
        "Cartera óptima - Fitness Lambda"
    )

    graficar_cartera(
        portfolio_sharpe,
        "Cartera óptima - Índice de Sharpe"
    )


# ============================================================
# 18. MENÚ PRINCIPAL
# ============================================================

def menu():

    while True:

        print("\n")
        print("=" * 60)
        print("             OPTIMIZACIÓN DE CARTERAS")
        print("=" * 60)

        print()
        print("1. Ejecutar con Fitness Lambda")
        print("2. Ejecutar con Índice de Sharpe")
        print("3. Comparar ambos métodos")
        print("0. Salir")

        print()
        print("=" * 60)

        option = input(
            "Seleccione una opción: "
        )

        # ----------------------------------------------------
        # OPCIÓN 1 - LAMBDA
        # ----------------------------------------------------

        if option == "1":

            print(
                "\nEjecutando algoritmo "
                "con Fitness Lambda...\n"
            )

            best_portfolio, best_fitness = (
                algoritmo_genetico(
                    fitness_lambda
                )
            )

            mostrar_resultados(
                "RESULTADO - FITNESS LAMBDA",
                best_portfolio,
                best_fitness
            )

            graficar_cartera(
                best_portfolio,
                "Cartera óptima - Fitness Lambda"
            )

        # ----------------------------------------------------
        # OPCIÓN 2 - SHARPE
        # ----------------------------------------------------

        elif option == "2":

            print(
                "\nEjecutando algoritmo "
                "con Índice de Sharpe...\n"
            )

            best_portfolio, best_fitness = (
                algoritmo_genetico(
                    fitness_sharpe
                )
            )

            mostrar_resultados(
                "RESULTADO - ÍNDICE DE SHARPE",
                best_portfolio,
                best_fitness
            )

            graficar_cartera(
                best_portfolio,
                "Cartera óptima - Índice de Sharpe"
            )

        # ----------------------------------------------------
        # OPCIÓN 3 - COMPARACIÓN
        # ----------------------------------------------------

        elif option == "3":

            comparar_metodos()

        # ----------------------------------------------------
        # OPCIÓN 0 - SALIR
        # ----------------------------------------------------

        elif option == "0":

            print("\nPrograma finalizado.")

            break

        else:

            print(
                "\nOpción inválida. "
                "Intente nuevamente."
            )


# ============================================================
# 19. OBTENER DATOS Y EJECUTAR
# ============================================================

print("\nObteniendo datos históricos...")

precios = obtener_precios(tickers)

retornos = calcular_rendimientos(precios)

# Rendimiento esperado de cada activo.
mu = retornos.mean().values

# Matriz de covarianzas.
cov_matrix = retornos.cov().values

print("\nDatos obtenidos correctamente.")

print("\n--- RENDIMIENTOS HISTÓRICOS ---")
print(retornos)

# ============================================================
# 20. INICIAR MENÚ
# ============================================================

menu()