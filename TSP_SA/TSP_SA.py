import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random 

from math import radians,cos,sin,sqrt

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

def total_fittness(route):
    total_distance = 0
    for i in range(0, len(route)):
        this_city = route[i]
        next_city = None
        if i + 1 < len(route):
            next_city = route[i + 1]
        else:
            next_city = route[0]
        total_distance += distance(this_city, next_city)

    fitness_value = 1/total_distance
    return fitness_value

def simulated_annealing(route, temperature, temperature_factor, stop_temperature):
    current_route = route[:]
    current_value = total_fittness(current_route)
    i = 0
    while temperature > stop_temperature:

        temperature *= temperature_factor
        i += 1
        current_cost = 1/current_value
        
        city1, city2 = np.random.randint(0,len(route),size=2)
        current_route[city1], current_route[city2] = current_route[city2], current_route[city1]

        next_value = total_fittness(current_route)
        delta_E = next_value - current_value
        if delta_E >= 0:
            current_value = next_value
        else:
            x = np.random.uniform()
            if x < np.exp(delta_E / temperature):
                current_value = next_value
            else:
                current_route[city1], current_route[city2] = current_route[city2], current_route[city1]

    return [current_route, current_cost]
        
def plotting(cities,ax):
    for i in range(0, len(cities)):
        first_city = cities[i]
        ax.plot(cities[i][2],cities[i][1],'ro')
        second_city = None
        if i + 1 < len(cities):
            second_city = cities[i + 1]
        else:
            second_city = cities[0]

        xpoints = [first_city[2],second_city[2]]
        ypoints = [first_city[1],second_city[1]]
        ax.plot(xpoints, ypoints,'b')

if __name__ == '__main__':

    fields = ['Name','Latitude','Longitude']
    df = pd.read_csv('US_States.csv', usecols=fields, skiprows=[2, 9])

    num_of_cities = 49
    cities = df.values.tolist()
    cities = cities[0:num_of_cities]

    # simulated_annealing
    result = simulated_annealing(route=cities, temperature=400, temperature_factor=0.99999, stop_temperature=1e-16)

    print('---------------Best Result---------------')
    print('Route:', '\n', np.array(result[0]), '\n')
    print('Distance:','\n', result[1], '\n')

    # creating csv file for gps apps
    result[0].append(result[0][0])
    df = pd.DataFrame(result[0], columns = ['Name','Latitude','Longitude'])
    tsp = df[["Latitude", "Longitude"]]
    tsp.to_csv('US_TSP_Solution.csv', index=False)

    # plot the result
    fig=plt.figure(figsize=(10,5))
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)
    plotting(cities,ax1)
    plotting(result[0],ax2)
    plt.show()