import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. ACTIVOS
# ============================================================

tickers = {
    "SP500": "SPY",
    "GOLD": "GLD",
    "YPF": "YPF",
    "COCA-COLA": "KO",
    "QQQ": "QQQ",
    "MERCADO LIBRE": "MELI",
    "NVIDIA": "NVDA",
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

    # En caso de que yfinance devuelva una sola columna.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

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

TAM_CARTERA = len(tickers)

TAM_POBLACION = 100
GENERACIONES = 100

PROB_MUTACION = 0.05
DESVIACION_MUTACION = 0.05

PROB_CROSSOVER = 0.8

# Coeficiente de aversión al riesgo.
<<<<<<< HEAD
LAMBDA = 0.5 
=======
LAMBDA = 3
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0

# Tasa libre de riesgo mensual.
RISK_FREE_RATE = 0.0

# ------------------------------------------------------------
# Restricciones de inversión
# ------------------------------------------------------------

# Mínimo y máximo permitido por activo.
MIN_INVERSION = 0.05
MAX_INVERSION = 0.30

if TAM_CARTERA * MIN_INVERSION > 1:
    raise ValueError(
        "La inversión mínima no permite distribuir el 100% del capital."
    )

if TAM_CARTERA * MAX_INVERSION < 1:
    raise ValueError(
        "La inversión máxima no permite distribuir el 100% del capital."
    )


# ============================================================
# 4. AJUSTAR CARTERA A LAS RESTRICCIONES
# ============================================================

def ajustar_restricciones(proporciones):
    """
    Ajusta una cartera para cumplir simultáneamente:

        MIN_INVERSION <= peso_i <= MAX_INVERSION

        sum(peso_i) = 1

    Se utiliza una redistribución iterativa del capital.
    """

    pesos = np.asarray(proporciones, dtype=float).copy()

    # Evitamos valores negativos.
    pesos = np.maximum(pesos, 0)

    # Si todos los pesos fueran cero, generamos una cartera nueva.
    if np.sum(pesos) == 0:
        return generar_cartera()

    # Primera normalización.
    pesos /= np.sum(pesos)

    # Aplicamos límites.
    pesos = np.clip(
        pesos,
        MIN_INVERSION,
        MAX_INVERSION
    )

    # Redistribuimos hasta conseguir suma 1.
    for _ in range(100):

        diferencia = 1.0 - np.sum(pesos)

        if abs(diferencia) < 1e-12:
            break

        if diferencia > 0:

            # Capital que todavía se puede agregar.
            capacidad = MAX_INVERSION - pesos

            capacidad_total = np.sum(capacidad)

            if capacidad_total <= 0:
                break

            pesos += diferencia * (
                capacidad / capacidad_total
            )

        else:

            # Capital que se debe quitar.
            capacidad = pesos - MIN_INVERSION

            capacidad_total = np.sum(capacidad)

            if capacidad_total <= 0:
                break

            pesos += diferencia * (
                capacidad / capacidad_total
            )

        pesos = np.clip(
            pesos,
            MIN_INVERSION,
            MAX_INVERSION
        )

    # Corrección numérica final.
    pesos /= np.sum(pesos)

    # Una última corrección por posibles errores de redondeo.
    diferencia = 1.0 - np.sum(pesos)

    if abs(diferencia) > 1e-12:

        if diferencia > 0:
            indices = np.where(
                pesos < MAX_INVERSION - 1e-12
            )[0]
        else:
            indices = np.where(
                pesos > MIN_INVERSION + 1e-12
            )[0]

        if len(indices) > 0:
            pesos[indices[0]] += diferencia

    return pesos


# ============================================================
# 5. GENERAR UNA CARTERA
# ============================================================

def generar_cartera():
    """
    Genera un individuo.

    Cada gen representa el porcentaje invertido
    en un activo.

    La cartera cumple las restricciones de inversión
    mínima y máxima.
    """

    # Comenzamos asignando el mínimo a todos.
    pesos = np.full(
        TAM_CARTERA,
        MIN_INVERSION
    )

    # Capital restante.
    restante = 1.0 - np.sum(pesos)

    # Capacidad restante de cada activo.
    capacidades = np.full(
        TAM_CARTERA,
        MAX_INVERSION - MIN_INVERSION
    )

    while restante > 1e-12:

        indices = np.where(
            capacidades > 1e-12
        )[0]

        if len(indices) == 0:
            break

        valores = np.random.random(
            len(indices)
        )

        valores /= np.sum(valores)

        asignacion = np.minimum(
            valores * restante,
            capacidades[indices]
        )

        pesos[indices] += asignacion
        capacidades[indices] -= asignacion

        restante = 1.0 - np.sum(pesos)

    return ajustar_restricciones(pesos)


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

    return np.dot(
        proporciones,
        mu
    )


# ============================================================
# 8. RIESGO DE LA CARTERA
# ============================================================

def riesgo_cartera(proporciones):

    varianza = (
        proporciones.T
        @ cov_matrix
        @ proporciones
    )

    return np.sqrt(
        max(varianza, 0)
    )


# ============================================================
# 9. FITNESS - LAMBDA
# ============================================================

def fitness_lambda(proporciones):

    expected_return = retorno_cartera(
        proporciones
    )

    riesgo = riesgo_cartera(
        proporciones
    )

    return (
        expected_return
        - LAMBDA * riesgo
    )


# ============================================================
# 10. FITNESS - ÍNDICE DE SHARPE
# ============================================================

def fitness_sharpe(proporciones):

    expected_return = retorno_cartera(
        proporciones
    )

    riesgo = riesgo_cartera(
        proporciones
    )

    # Evitamos división por cero.
    if riesgo == 0:
        return 0

    return (
        expected_return
        - RISK_FREE_RATE
    ) / riesgo


# ============================================================
# 11. SELECCIÓN
# ============================================================

def seleccion_truncamiento(
    poblacion,
    fitness_function
):

    fitness_values = np.array([
        fitness_function(individual)
        for individual in poblacion
    ])

    # Ordenamos de mejor a peor.
    indexes = np.argsort(
        fitness_values
    )[::-1]

    # Seleccionamos la mitad superior.
    selected = poblacion[
        indexes[:TAM_POBLACION // 2]
    ]

    return selected


# ============================================================
# 12. CRUCE
# ============================================================

def crossover(p1, p2):

    if np.random.random() > PROB_CROSSOVER:
        return p1.copy()

    alpha = np.random.random()

    hijo = (
        alpha * p1
        + (1 - alpha) * p2
    )

    # Aplicamos las restricciones.
    hijo = ajustar_restricciones(
        hijo
    )

    return hijo


# ============================================================
# 13. MUTACIÓN
# ============================================================

def mutacion(individuo):

    individuo = individuo.copy()

    for i in range(TAM_CARTERA):

        if np.random.random() < PROB_MUTACION:

            individuo[i] += np.random.normal(
                0,
                DESVIACION_MUTACION
            )

    # Aplicamos las restricciones.
    individuo = ajustar_restricciones(
        individuo
    )

    return individuo


# ============================================================
# 14. CREAR NUEVA POBLACIÓN
# ============================================================

def crear_nueva_poblacion(
    poblacion_selec
):

    nueva_poblacion = []

    while len(nueva_poblacion) < TAM_POBLACION:

        parent1 = poblacion_selec[
            np.random.randint(
                len(poblacion_selec)
            )
        ]

        parent2 = poblacion_selec[
            np.random.randint(
                len(poblacion_selec)
            )
        ]

        child = crossover(
            parent1,
            parent2
        )

        child = mutacion(
            child
        )

        nueva_poblacion.append(
            child
        )

    return np.array(
        nueva_poblacion
    )


# ============================================================
# 15. ALGORITMO GENÉTICO
# ============================================================

def algoritmo_genetico(
    fitness_function,
    poblacion_inicial=None
):
    """
    Ejecuta el algoritmo genético.

    Además de la mejor cartera, devuelve el historial
    de poblaciones para poder visualizar su evolución.
    """

    if poblacion_inicial is None:

        population = generar_poblacion()

    else:

        population = (
            poblacion_inicial.copy()
        )

    best_individual = None
    best_fitness = -np.inf

    # Historial completo de las poblaciones.
    historial_poblaciones = []

    # Historial del mejor individuo de cada generación.
    historial_mejores = []

    for generation in range(
        GENERACIONES
    ):

        # ----------------------------------------------------
        # Guardamos la población actual.
        # ----------------------------------------------------

        historial_poblaciones.append(
            population.copy()
        )

        # ----------------------------------------------------
        # Evaluamos la población.
        # ----------------------------------------------------

        fitness_values = np.array([
            fitness_function(individual)
            for individual in population
        ])

        # Mejor individuo de la generación.
        best_index = np.argmax(
            fitness_values
        )

        current_best = (
            population[
                best_index
            ].copy()
        )

        current_fitness = (
            fitness_values[
                best_index
            ]
        )

        historial_mejores.append(
            current_best.copy()
        )

        # ----------------------------------------------------
        # Mejor individuo global.
        # ----------------------------------------------------

        if current_fitness > best_fitness:

            best_fitness = (
                current_fitness
            )

            best_individual = (
                current_best.copy()
            )

        # ----------------------------------------------------
        # Selección.
        # ----------------------------------------------------

        selected = (
            seleccion_truncamiento(
                population,
                fitness_function
            )
        )

        # ----------------------------------------------------
        # Cruce + mutación.
        # ----------------------------------------------------

        population = (
            crear_nueva_poblacion(
                selected
            )
        )

        print(
            f"Generación {generation + 1}: "
            f"Fitness = {current_fitness:.6f}"
        )

    return (
        best_individual,
        best_fitness,
        historial_poblaciones,
        historial_mejores
    )


# ============================================================
# 16. MOSTRAR RESULTADOS
# ============================================================

def mostrar_resultados(
    titulo,
    portfolio,
    fitness_value
):

    expected_return = (
        retorno_cartera(
            portfolio
        )
    )

    risk = (
        riesgo_cartera(
            portfolio
        )
    )

    sharpe = (
        fitness_sharpe(
            portfolio
        )
    )

    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)

    print(
        "\nDistribución de la cartera:"
    )

    for asset, weight in zip(
        retornos.columns,
        portfolio
    ):

        print(
            f"{asset:<20}: "
            f"{weight:>8.2%}"
        )

    print(
        "\nSuma de pesos:"
    )

    print(
        f"{np.sum(portfolio):.6f}"
    )

    print(
        "\nRendimiento esperado:"
    )

    print(
        f"{expected_return:.6f}"
    )

    print(
        "\nRiesgo:"
    )

    print(
        f"{risk:.6f}"
    )

    print(
        "\nÍndice de Sharpe:"
    )

    print(
        f"{sharpe:.6f}"
    )

    print(
        "\nFitness:"
    )

    print(
        f"{fitness_value:.6f}"
    )


# ============================================================
# 17. GRAFICAR CARTERA
# ============================================================

def graficar_cartera(
    mejor_cartera,
    titulo="Distribución de la cartera óptima"
):

    rendimiento_mensual = (
        retorno_cartera(
            mejor_cartera
        )
    )

    riesgo_mensual = (
        riesgo_cartera(
            mejor_cartera
        )
    )

    # Tasa anual simple.
    rendimiento_anual_tda = (
        rendimiento_mensual * 12
    )

    # Tasa efectiva anual.
    rendimiento_anual_tea = (
        (1 + rendimiento_mensual) ** 12
        - 1
    )

    # Anualización de la volatilidad.
    riesgo_anual = (
        riesgo_mensual * np.sqrt(12)
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.pie(
        mejor_cartera,
        labels=retornos.columns,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        titulo
    )

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
<<<<<<< HEAD
        ha="left",
=======
        ha="center",
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
        fontsize=11
    )

    plt.tight_layout(
        rect=[
            0,
            0.08,
            1,
            1
        ]
    )

    plt.show()
<<<<<<< HEAD

# ============================================================
# 19. PREPARAR DATOS PARA EL GRÁFICO DE EVOLUCIÓN
# ============================================================


def convertir_historial_xy(
    historial_poblaciones
):
    """
    Convierte cada población del historial a puntos
    (riesgo, rendimiento).
    """

    historial_xy = []

=======


# ============================================================
# 18. PREPARAR DATOS PARA EL GRÁFICO DE EVOLUCIÓN
# ============================================================

def convertir_historial_xy(
    historial_poblaciones
):
    """
    Convierte cada población del historial a puntos
    (riesgo, rendimiento).
    """

    historial_xy = []

>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
    for poblacion in (
        historial_poblaciones
    ):

        puntos = []

        for individuo in (
            poblacion
        ):

            riesgo = (
                riesgo_cartera(
                    individuo
                )
            )

            rendimiento = (
                retorno_cartera(
                    individuo
                )
            )

            puntos.append([
                riesgo * 100,
                rendimiento * 100
            ])

        historial_xy.append(
            np.array(puntos)
        )

    return historial_xy


# ============================================================
<<<<<<< HEAD
# 20. GRAFICAR EVOLUCIÓN DE LAS GENERACIONES
=======
# 19. GRAFICAR EVOLUCIÓN DE LAS GENERACIONES
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
# ============================================================

def graficar_evolucion(
    historial_poblaciones,
    fitness_function,
    titulo
):
<<<<<<< HEAD
    """
    Animación de la evolución del algoritmo genético
    superpuesta sobre la frontera eficiente de Markowitz.
    """
=======
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0

    historial_xy = convertir_historial_xy(
        historial_poblaciones
    )

<<<<<<< HEAD
    # --------------------------------------------------------
    # Calculamos la frontera una sola vez.
    # Lambda NO interviene en este cálculo.
    # --------------------------------------------------------

=======
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.set_xlabel(
        "Volatilidad (%)"
    )

    ax.set_ylabel(
        "Rendimiento esperado (%)"
    )

    ax.set_title(
        titulo
    )

    ax.grid(
        True,
        alpha=0.3
    )

    # ========================================================
<<<<<<< HEAD
    # RANGO DE LOS EJES
=======
    # FIJAR RANGO DE LOS EJES
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
    # ========================================================

    todos_los_puntos = np.vstack(
        historial_xy
    )

<<<<<<< HEAD
    puntos_x = np.concatenate([
        todos_los_puntos[:, 0],
    ])

    puntos_y = np.concatenate([
        todos_los_puntos[:, 1],
    ])

    x_min = np.min(puntos_x)
    x_max = np.max(puntos_x)
    y_min = np.min(puntos_y)
    y_max = np.max(puntos_y)

    margen_x = (x_max - x_min) * 0.10
    margen_y = (y_max - y_min) * 0.10
=======
    x_min = np.min(
        todos_los_puntos[:, 0]
    )

    x_max = np.max(
        todos_los_puntos[:, 0]
    )

    y_min = np.min(
        todos_los_puntos[:, 1]
    )

    y_max = np.max(
        todos_los_puntos[:, 1]
    )

    margen_x = (
        (x_max - x_min) * 0.10
    )

    margen_y = (
        (y_max - y_min) * 0.10
    )
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0

    if margen_x == 0:
        margen_x = 1

    if margen_y == 0:
        margen_y = 1

    ax.set_xlim(
        x_min - margen_x,
        x_max + margen_x
    )

    ax.set_ylim(
        y_min - margen_y,
        y_max + margen_y
    )

<<<<<<< HEAD
    # ========================================================
    # FRONTERA DE MARKOWITZ
    # ========================================================

    ax.plot(

        linewidth=2.5,
        label="Frontera eficiente de Markowitz"
    )

    # ========================================================
    # POBLACIÓN
    # ========================================================
=======
    # --------------------------------------------------------
    # Población inicial.
    # --------------------------------------------------------
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0

    scatter = ax.scatter(
        historial_xy[0][:, 0],
        historial_xy[0][:, 1],
        s=25,
        alpha=0.6,
        label="Población"
    )

<<<<<<< HEAD
    # ========================================================
    # MEJOR INDIVIDUO
    # ========================================================
=======
    # --------------------------------------------------------
    # Mejor individuo.
    # --------------------------------------------------------
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0

    mejor_scatter = ax.scatter(
        [],
        [],
        s=120,
        marker="*",
        label="Mejor individuo"
    )

<<<<<<< HEAD
=======
    # --------------------------------------------------------
    # Texto de generación.
    # --------------------------------------------------------

>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
    texto = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top"
    )

    ax.legend()

<<<<<<< HEAD
    # ========================================================
    # ACTUALIZACIÓN DE LA ANIMACIÓN
    # ========================================================

    def actualizar(generacion):

        puntos = historial_xy[generacion]
=======
    # --------------------------------------------------------
    # Función de actualización.
    # --------------------------------------------------------

    def actualizar(generacion):

        puntos = (
            historial_xy[
                generacion
            ]
        )
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0

        scatter.set_offsets(
            puntos
        )

<<<<<<< HEAD
        fitness_values = [
            fitness_function(individual)
            for individual in historial_poblaciones[generacion]
        ]

=======
        # Fitness de cada individuo.
        fitness_values = [
            fitness_function(
                individuo
            )
            for individuo in (
                historial_poblaciones[
                    generacion
                ]
            )
        ]

        # Mejor individuo.
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
        mejor_index = np.argmax(
            fitness_values
        )

        mejor_individuo = (
            historial_poblaciones[
                generacion
            ][
                mejor_index
            ]
        )

        mejor_punto = [
            riesgo_cartera(
                mejor_individuo
            ) * 100,

            retorno_cartera(
                mejor_individuo
            ) * 100
        ]

<<<<<<< HEAD
        # Mostramos el mejor individuo en todas las generaciones.
        mejor_scatter.set_offsets(
            [mejor_punto]
        )
=======
        if generacion == len(historial_poblaciones) - 1:
            mejor_scatter.set_offsets([mejor_punto])
            mejor_scatter.set_visible(True)
        else:
            mejor_scatter.set_visible(False)
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0

        texto.set_text(
            f"Generación: "
            f"{generacion + 1} / "
            f"{len(historial_poblaciones)}"
        )

        return (
            scatter,
            mejor_scatter,
            texto
        )

<<<<<<< HEAD
    animacion = FuncAnimation(
        fig,
        actualizar,
        frames=len(historial_poblaciones),
=======
    # --------------------------------------------------------
    # Crear animación.
    # --------------------------------------------------------

    animacion = FuncAnimation(
        fig,
        actualizar,
        frames=len(
            historial_poblaciones
        ),
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
        interval=150,
        repeat=False
    )

<<<<<<< HEAD
    plt.tight_layout()
=======
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
    plt.show()

    return animacion


# ============================================================
<<<<<<< HEAD
# 21. COMPARAR MÉTODOS
=======
# 20. COMPARAR MÉTODOS
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
# ============================================================

def comparar_metodos():

    print("\n")
    print("=" * 60)
    print(
        "              COMPARACIÓN DE MÉTODOS"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Misma población inicial para ambos métodos.
    # --------------------------------------------------------

    poblacion_inicial = (
        generar_poblacion()
    )

    # --------------------------------------------------------
    # FITNESS LAMBDA
    # --------------------------------------------------------

    print(
        "\nEjecutando Fitness Lambda...\n"
    )

    (
        portfolio_lambda,
        fitness_lambda_value,
        historial_lambda,
        mejores_lambda
    ) = algoritmo_genetico(
        fitness_lambda,
        poblacion_inicial
    )

    # --------------------------------------------------------
    # FITNESS SHARPE
    # --------------------------------------------------------

    print(
        "\nEjecutando Índice de Sharpe...\n"
    )

    (
        portfolio_sharpe,
        fitness_sharpe_value,
        historial_sharpe,
        mejores_sharpe
    ) = algoritmo_genetico(
        fitness_sharpe,
        poblacion_inicial
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

    return_lambda = (
        retorno_cartera(
            portfolio_lambda
        )
    )

    risk_lambda = (
        riesgo_cartera(
            portfolio_lambda
        )
    )

    sharpe_lambda = (
        fitness_sharpe(
            portfolio_lambda
        )
    )

    return_sharpe = (
        retorno_cartera(
            portfolio_sharpe
        )
    )

    risk_sharpe = (
        riesgo_cartera(
            portfolio_sharpe
        )
    )

    sharpe_sharpe = (
        fitness_sharpe(
            portfolio_sharpe
        )
    )

    # --------------------------------------------------------
    # TABLA COMPARATIVA
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "                         COMPARACIÓN"
    )
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
    print(
        "                    DISTRIBUCIÓN DE CARTERAS"
    )
    print("=" * 70)

    print(
        f"{'Activo':<20}"
        f"{'Lambda':>20}"
        f"{'Sharpe':>20}"
    )

    print("-" * 70)

    for (
        asset,
        weight_lambda,
        weight_sharpe
    ) in zip(
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
    # GRÁFICOS DE CARTERA
    # --------------------------------------------------------

    graficar_cartera(
        portfolio_lambda,
        "Cartera óptima - Fitness Lambda"
    )

    graficar_cartera(
        portfolio_sharpe,
        "Cartera óptima - Índice de Sharpe"
    )

    # --------------------------------------------------------
    # EVOLUCIÓN DE LAS GENERACIONES
    # --------------------------------------------------------

    graficar_evolucion(
        historial_lambda,
        fitness_lambda,
        "Evolución de las generaciones - Fitness Lambda"
    )

    graficar_evolucion(
        historial_sharpe,
        fitness_sharpe,
        "Evolución de las generaciones - Índice de Sharpe"
    )


<<<<<<< HEAD

# ============================================================
# 22. COMPARAR DISTINTOS VALORES DE LAMBDA
# ============================================================

def comparar_lambdas():
    """
    Ejecuta el algoritmo genético para distintos valores de
    lambda y superpone la evolución de cada uno sobre la misma
    frontera eficiente de Markowitz.

    Se utiliza la misma población inicial para que la comparación
    sea más consistente.
    """

    valores_lambda = [
        0.0,
        0.25,
        0.375,
        0.5,
        1.0,
        2.0,
    ]

    print("\n")
    print("=" * 70)
    print("              COMPARACIÓN DE VALORES DE LAMBDA")
    print("=" * 70)

    print(
        "\nValores a probar:",
        valores_lambda
    )
    
    # --------------------------------------------------------
    # Misma población inicial para todos los lambda.
    # --------------------------------------------------------

    poblacion_inicial = generar_poblacion()

    resultados = []

    # Guardamos las evoluciones para el gráfico conjunto.
    evoluciones = []

    # --------------------------------------------------------
    # Ejecutamos un AG por cada lambda.
    # --------------------------------------------------------

    for lambda_actual in valores_lambda:

        print(
            f"\n{'-' * 70}\n"
            f"Ejecutando lambda = {lambda_actual}\n"
            f"{'-' * 70}"
        )

        # Cambiamos temporalmente el lambda global.
        global LAMBDA
        LAMBDA = lambda_actual

        (
            cartera,
            fitness_value,
            historial_poblaciones,
            historial_mejores
        ) = algoritmo_genetico(
            fitness_lambda,
            poblacion_inicial
        )

        rendimiento = retorno_cartera(
            cartera
        )

        riesgo = riesgo_cartera(
            cartera
        )

        sharpe = fitness_sharpe(
            cartera
        )

        resultados.append({
            "lambda": lambda_actual,
            "rendimiento": rendimiento,
            "riesgo": riesgo,
            "sharpe": sharpe,
            "fitness": fitness_value,
            "cartera": cartera
        })

        evoluciones.append(
        (
            lambda_actual,
            historial_mejores,
            cartera
        )
    )
    # ========================================================
    # TABLA DE RESULTADOS
    # ========================================================

    print("\n")
    print("=" * 85)
    print("                         RESULTADOS")
    print("=" * 85)

    print(
        f"{'Lambda':>10}"
        f"{'Rendimiento':>20}"
        f"{'Riesgo':>20}"
        f"{'Sharpe':>15}"
        f"{'Fitness':>20}"
    )

    print("-" * 85)

    for resultado in resultados:

        print(
            f"{resultado['lambda']:>10.2f}"
            f"{resultado['rendimiento']:>19.6f}"
            f"{resultado['riesgo']:>20.6f}"
            f"{resultado['sharpe']:>15.6f}"
            f"{resultado['fitness']:>20.6f}"
        )

    print("=" * 85)

    # ========================================================
    # GRÁFICO COMPARATIVO DE EVOLUCIONES
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    for lambda_actual, mejores, mejor_global in evoluciones:

        # ========================================================
        # MEJOR INDIVIDUO DE CADA GENERACIÓN
        # ========================================================

        riesgos_mejores = [
            riesgo_cartera(individuo) * 100
            for individuo in mejores
        ]

        rendimientos_mejores = [
            retorno_cartera(individuo) * 100
            for individuo in mejores
        ]

        # --------------------------------------------------------
        # Puntos aislados de la evolución
        # --------------------------------------------------------

        ax.scatter(
            riesgos_mejores,
            rendimientos_mejores,
            s=15,
            alpha=0.5
            )

        # ========================================================
        # MEJOR INDIVIDUO GLOBAL DE TODA LA EJECUCIÓN
        # ========================================================

        riesgo_global = (
            riesgo_cartera(mejor_global) * 100
        )

        rendimiento_global = (
            retorno_cartera(mejor_global) * 100
        )

        ax.scatter(
            riesgo_global,
            rendimiento_global,
            s=150,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label=f"λ = {lambda_actual:g}"
        )
        
    ax.set_xlabel(
        "Volatilidad (%)"
    )

    ax.set_ylabel(
        "Rendimiento esperado (%)"
    )

    ax.set_title(
        "Evolución del algoritmo genético para distintos valores de λ"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        fontsize=9
    )

    plt.tight_layout()
    plt.show()

    return resultados


# ============================================================
# 23. MENÚ PRINCIPAL
=======
# ============================================================
# 21. MENÚ PRINCIPAL
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
# ============================================================

def menu():

    while True:

        print("\n")
        print("=" * 60)
        print(
            "             OPTIMIZACIÓN DE CARTERAS"
        )
        print("=" * 60)

        print()
        print(
            "1. Ejecutar con Fitness Lambda"
        )
        print(
            "2. Ejecutar con Índice de Sharpe"
        )
        print(
            "3. Comparar ambos métodos"
        )
        print(
<<<<<<< HEAD
            "4. Comparar distintos valores de Lambda"
        )
        print(
=======
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
            "0. Salir"
        )

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

            (
                best_portfolio,
                best_fitness,
                historial_poblaciones,
                historial_mejores
            ) = algoritmo_genetico(
                fitness_lambda
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

            graficar_evolucion(
                historial_poblaciones,
                fitness_lambda,
                "Evolución de las generaciones - Fitness Lambda"
            )

        # ----------------------------------------------------
        # OPCIÓN 2 - SHARPE
        # ----------------------------------------------------

        elif option == "2":

            print(
                "\nEjecutando algoritmo "
                "con Índice de Sharpe...\n"
            )

            (
                best_portfolio,
                best_fitness,
                historial_poblaciones,
                historial_mejores
            ) = algoritmo_genetico(
                fitness_sharpe
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

            graficar_evolucion(
                historial_poblaciones,
                fitness_sharpe,
                "Evolución de las generaciones - Índice de Sharpe"
            )

        # ----------------------------------------------------
        # OPCIÓN 3 - COMPARACIÓN
        # ----------------------------------------------------

        elif option == "3":

            comparar_metodos()

        # ----------------------------------------------------
<<<<<<< HEAD
        # OPCIÓN 4 - COMPARACIÓN DE LAMBDA
        # ----------------------------------------------------

        elif option == "4":

            comparar_lambdas()

        # ----------------------------------------------------
=======
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
        # OPCIÓN 0 - SALIR
        # ----------------------------------------------------

        elif option == "0":

            print(
                "\nPrograma finalizado."
            )

            break

        else:

            print(
                "\nOpción inválida. "
                "Intente nuevamente."
            )


# ============================================================
<<<<<<< HEAD
# 24. OBTENER DATOS Y EJECUTAR
=======
# 22. OBTENER DATOS Y EJECUTAR
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
# ============================================================

print(
    "\nObteniendo datos históricos..."
)

precios = obtener_precios(
    tickers
)

retornos = calcular_rendimientos(
    precios
)

# Rendimiento esperado de cada activo.
mu = retornos.mean().values

# Matriz de covarianzas.
cov_matrix = retornos.cov().values

print(
    "\nDatos obtenidos correctamente."
)

print(
    "\n--- RENDIMIENTOS HISTÓRICOS ---"
)

print(
    retornos
)


# ============================================================
<<<<<<< HEAD
# 25. INICIAR MENÚ
=======
# 23. INICIAR MENÚ
>>>>>>> 7deebe7509b355a9d9cf113f54d056a9cead0df0
# ============================================================

menu()
