import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

from math import radians,cos,sin,sqrt


def initial_population(genes, population_size):
    population = []
    for i in range(population_size):
        individual = random.sample(genes,len(genes))
        population.append(individual)
    return population

def cross_over(parent1,parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def mutation(individual, mutation_rate):
    for swapped in range(len(individual)):
        if(random.random() < mutation_rate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def breeding(population,mutation_rate):
    children = []
    for i in range(0,len(population)//2):
        parent1 = population[i]
        parent2 = population[-i]
        child = cross_over(parent1,parent2)
        children.append(child)
    for i in range(len(children)):
        children[i] = mutation(children[i], mutation_rate)
    return children

def latitude_longitude_to_x_y(city):
    lat = radians(city[1])
    lon = radians(city[2])
    x = cos(lon)*cos(lat)*6371
    y = cos(lon)*sin(lat)*6371
    return x, y
    
def distance(city1,city2):
    city1_x, city1_y = latitude_longitude_to_x_y(city1)
    city2_x, city2_y = latitude_longitude_to_x_y(city2)
    x_distance = abs(city1_x - city2_x)
    y_distance = abs(city1_y - city2_y)
    distance = sqrt((x_distance ** 2) + (y_distance ** 2))
    return distance
    
def fitness(individual):
    total_distance = 0
    for i in range(0, len(individual)):
        this_city = individual[i]
        next_city = None
        if i + 1 < len(individual):
            next_city = individual[i + 1]
        else:
            next_city = individual[0]
        total_distance += distance(this_city, next_city)

    fitness_value = 1/total_distance
    return fitness_value

def fittest(population):
    best = 0
    best_individual = []

    for individual in population:
        individual_fitness = fitness(individual)
        if individual_fitness > best:
            best = individual_fitness
            best_individual = individual
    
    return [best_individual,best]


def selection(population_with_fitness, population_size, elite_size):
    next_gen = []
    elites = []

    for i in range(elite_size):
        elite_with_fitness = population_with_fitness.pop()
        elites.append(elite_with_fitness[0])

    population = []
    fitnesses = []
    for individual_with_fitness in population_with_fitness:
        population.append(individual_with_fitness[0])
        fitnesses.append(individual_with_fitness[1])

    probabilities = []
    sum_fitness=sum(fitnesses)
    for i in range(len(fitnesses)):
        probabilities.append( fitnesses[i]/sum_fitness)

    next_gen = random.choices(population, weights=probabilities, k=population_size-elite_size)
    next_gen.extend(elites)
    return next_gen


def next_generation(population, population_size, elite_size):
    population_with_fitnesses = []

    for individual in population:
        population_with_fitnesses.append((individual, fitness(individual) )) 

    sorted_population = sorted(population_with_fitnesses, key=lambda x: x[1])

    next_gen = selection(sorted_population, population_size, elite_size)

    return next_gen


def genetic_algorithm(genes, population_size, num_of_generations, mutation_rate, elite_size):
    population = initial_population(genes, population_size)

    # 0-route 1-fitness 2-generation number
    alltime_fittest = [[],0,0]

    for generation_num in range(num_of_generations):
        children = breeding(population, mutation_rate)

        for child in children:
            population.append(child)

        population = next_generation(population, population_size, elite_size)

        generation_fittest = fittest(population)
        generation_fittest.append(generation_num)
        if generation_fittest[1] > alltime_fittest[1]:
            alltime_fittest = generation_fittest

    return alltime_fittest


if __name__ == '__main__':

    fields = ['Name','Latitude','Longitude']
    df = pd.read_csv('US_States.csv', usecols=fields, skiprows=[2, 9])

    num_of_cities = 49
    cities = df.values.tolist()
    cities = cities[0:num_of_cities]

    result = genetic_algorithm(genes=cities, population_size=500, num_of_generations=6000, mutation_rate= 0.01, elite_size=100)

    print('---------------Fittest Result---------------')
    print('Route:', '\n', np.array(result[0]), '\n')
    print('Distance:','\n', 1/result[1], '\n')
    print('Geneation Number:', '\n', result[2], '\n')

    result[0].append(result[0][0])
    df = pd.DataFrame(result[0], columns = ['Name','Latitude','Longitude'])
    tsp = df[["Latitude", "Longitude"]]

    tsp.to_csv('US_TSP_Solution.csv', index=False)

    df.plot(kind = 'line', x = 'Longitude', y = 'Latitude')
    plt.show()