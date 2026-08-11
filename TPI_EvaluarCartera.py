import numpy as np
import pandas as pd


# ============================================================
# 1. DATOS DE LOS ACTIVOS
# ============================================================

# Ejemplo: rendimientos históricos de 5 activos
# Cada columna representa un activo
returns = pd.DataFrame({
    "YPF": [...],
    "AL30": [...],
    "APPLE": [...],
    "ARCOR": [...],
    "FCI": [...]
})


# ============================================================
# 2. PARÁMETROS DEL MODELO
# ============================================================

NUM_ASSETS = returns.shape[1]
POPULATION_SIZE = 100
GENERATIONS = 500

MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8

LAMBDA = 0.5

# Tasa libre de riesgo, necesaria si usamos Sharpe
RISK_FREE_RATE = 0.0


# ============================================================
# 3. DATOS ESTADÍSTICOS
# ============================================================

# Rendimiento esperado de cada activo
mu = returns.mean().values

# Matriz de covarianzas
cov_matrix = returns.cov().values


# ============================================================
# 4. GENERAR UNA CARTERA
# ============================================================

def generate_portfolio():
    """
    Genera un individuo.
    Cada gen representa el porcentaje invertido
    en un activo.
    """

    weights = np.random.random(NUM_ASSETS)

    # Normalizamos para que la suma sea 1
    weights /= np.sum(weights)

    return weights


# ============================================================
# 5. GENERAR POBLACIÓN
# ============================================================

def generate_population():
    return np.array([
        generate_portfolio()
        for _ in range(POPULATION_SIZE)
    ])


# ============================================================
# 6. RENDIMIENTO DE LA CARTERA
# ============================================================

def portfolio_return(weights):
    return np.dot(weights, mu)


# ============================================================
# 7. RIESGO DE LA CARTERA
# ============================================================

def portfolio_risk(weights):

    variance = weights.T @ cov_matrix @ weights

    return np.sqrt(variance)


# ============================================================
# 8. FITNESS
# ============================================================

def fitness(weights):

    expected_return = portfolio_return(weights)
    risk = portfolio_risk(weights)

    return expected_return - LAMBDA * risk


# ============================================================
# 9. SELECCIÓN
# ============================================================

def selection(population):

    fitness_values = np.array([
        fitness(individual)
        for individual in population
    ])

    # Ordenamos de mejor a peor
    indexes = np.argsort(fitness_values)[::-1]

    # Seleccionamos los mejores
    selected = population[indexes[:POPULATION_SIZE // 2]]

    return selected


# ============================================================
# 10. CRUCE
# ============================================================

def crossover(parent1, parent2):

    if np.random.random() > CROSSOVER_RATE:
        return parent1.copy()

    alpha = np.random.random()

    child = alpha * parent1 + (1 - alpha) * parent2

    # Nos aseguramos de que los pesos sumen 1
    child /= np.sum(child)

    return child


# ============================================================
# 11. MUTACIÓN
# ============================================================

def mutation(individual):

    for i in range(NUM_ASSETS):

        if np.random.random() < MUTATION_RATE:

            # Pequeña modificación del peso
            individual[i] += np.random.normal(0, 0.05)

    # Evitamos pesos negativos
    individual = np.maximum(individual, 0)

    # Volvemos a normalizar
    if np.sum(individual) > 0:
        individual /= np.sum(individual)

    return individual


# ============================================================
# 12. CREAR NUEVA POBLACIÓN
# ============================================================

def create_new_population(selected):

    new_population = []

    while len(new_population) < POPULATION_SIZE:

        parent1 = selected[
            np.random.randint(len(selected))
        ]

        parent2 = selected[
            np.random.randint(len(selected))
        ]

        child = crossover(parent1, parent2)

        child = mutation(child)

        new_population.append(child)

    return np.array(new_population)


# ============================================================
# 13. ALGORITMO GENÉTICO
# ============================================================

def genetic_algorithm():

    population = generate_population()

    best_individual = None
    best_fitness = -np.inf

    for generation in range(GENERATIONS):

        # Evaluamos la población
        fitness_values = np.array([
            fitness(individual)
            for individual in population
        ])

        # Mejor individuo de esta generación
        best_index = np.argmax(fitness_values)

        current_best = population[best_index]
        current_fitness = fitness_values[best_index]

        # Guardamos el mejor global
        if current_fitness > best_fitness:

            best_fitness = current_fitness
            best_individual = current_best.copy()

        # Selección
        selected = selection(population)

        # Cruce + mutación
        population = create_new_population(selected)

        print(
            f"Generación {generation + 1}: "
            f"Fitness = {current_fitness:.6f}"
        )

    return best_individual, best_fitness


# ============================================================
# 14. EJECUCIÓN
# ============================================================

best_portfolio, best_fitness = genetic_algorithm()


# ============================================================
# 15. RESULTADOS
# ============================================================

print("\n--- MEJOR CARTERA ---")

for asset, weight in zip(returns.columns, best_portfolio):

    print(f"{asset}: {weight:.2%}")


print("\nRendimiento esperado:",
      portfolio_return(best_portfolio))

print("Riesgo:",
      portfolio_risk(best_portfolio))

print("Fitness:",
      best_fitness)