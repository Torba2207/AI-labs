from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1
mutation_rate=0.1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)

def selection(population,items, knapsack_max_capacity,n_selection):
    fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    total_fitness = sum(fitnesses)
    selection_probabilities = [fit / total_fitness for fit in fitnesses]
    selected_individuals = [None] * n_selection
    for i in range(n_selection):
        selected_individuals[i] = population[np.random.choice(len(population), p=selection_probabilities)]
    return selected_individuals

#def create_next_generation(population, n_elite):
#    next_generation = []

    # Step 1: Elite selection (keep the best individuals from the current generation)
#    next_generation.extend(population[:n_elite])

    # Step 2: Crossover (create offspring from parent pairs)
#    while len(next_generation) < len(population):
#        parent1, parent2 = random.sample(population, 2)
#        crossover_point = random.randint(1, len(parent1) - 1)
#        child1 = parent1[:crossover_point] + parent2[crossover_point:]
#        child2 = parent2[:crossover_point] + parent1[crossover_point:]
#        next_generation.extend([child1, child2])
#
    # Ensure the population size is maintained even if n_elite and n_offspring don't sum up to the total population size
#    next_generation = next_generation[:len(population)]
#
#    return next_generation


def eliteChoice(n_elite, population, items, knapsack_max_capacity):
#    population_fitness_indices = np.argsort([fitness(items, knapsack_max_capacity, individual) for individual in population])[::-1]
#    return [population[population_fitness_indices[i]] for i in range(n_elite)]
    old_population=population
    old_population.sort(key=lambda ind: fitness(items, knapsack_max_capacity, ind), reverse=True)
    elite=old_population[:n_elite]
    return elite

def crossover(parent1, parent2):
    crossover_point=1
    if(len(parent1)<len(parent2)):
        crossover_point = random.randint(1,len(parent1))
    else:
        crossover_point = random.randint(1, len(parent2))
    child1 = parent1[crossover_point:] + parent2[:crossover_point]
    child2 = parent2[crossover_point:] + parent1[:crossover_point]
    return child1, child2


def mutate_population(population,mut_rate):
    for individual in population:
        for i in range(len(individual)):
            prok=random.random()
            if(prok<mut_rate):
                individual[i] = not individual[i]


#def update_population_with_elitism(old_population, new_population, n_elite):
    # Sort both old and new populations based on fitness
#    old_population.sort(key=lambda ind: fitness(items, knapsack_max_capacity, ind), reverse=True)
#    new_population.sort(key=lambda ind: fitness(items, knapsack_max_capacity, ind), reverse=True)

    # Select the elite individuals from the old population
#    elite = old_population[:n_elite]

    # Replace the weakest individuals in the old population with the new population
#    old_population[-n_elite:] = new_population[:n_elite]

    # Merge old population with new population
#    updated_population = old_population + new_population[n_elite:]

#    return updated_population


for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm


    # Step 1: Calculate fitness for each individual in the population


    # Step 2: Roulette wheel selection

    elite = eliteChoice(n_elite, population, items, knapsack_max_capacity)
    selected_parents = selection(population,items, knapsack_max_capacity,n_selection)

    # Step 3: Create next generation

    next_generation = []

    while (len(next_generation)) < len(selected_parents):
        parent1, parent2 = random.sample(selected_parents, 2)
        child1, child2 = crossover(parent1, parent2)
        #next_generation=next_generation+child1+child2
        #next_generation=next_generation+child1
        #next_generation=next_generation+child2
        next_generation.extend([child1, child2])
    # Step 4: Mutation
    mutate_population(next_generation, mutation_rate)

    #Step 5: Refill
    new_population=next_generation
    new_population=new_population+elite
    refill_population=selection(population,items, knapsack_max_capacity,population_size-len(next_generation))

    population=new_population+refill_population






    # Step 5: Update population with elitism
#    population = update_population_with_elitism(population_history[-1], population, n_elite)


    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 100
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
