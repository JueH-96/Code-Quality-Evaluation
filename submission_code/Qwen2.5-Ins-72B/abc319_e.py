# YOUR CODE HERE
import sys

def next_departure_time(current_time, period):
    return (current_time + period - 1) // period * period

def min_time_to_reach_aoki(n, x, y, bus_info, queries):
    # Precompute the minimum time to reach each bus stop from the house
    min_time_to_bus_stop = [0] * (n + 1)
    min_time_to_bus_stop[1] = x
    
    for i in range(1, n):
        min_time_to_bus_stop[i + 1] = next_departure_time(min_time_to_bus_stop[i], bus_info[i - 1][0]) + bus_info[i - 1][1]
    
    # Add the time to walk from the last bus stop to Aoki's house
    min_time_to_aoki = min_time_to_bus_stop[n] + y
    
    results = []
    for q in queries:
        current_time = q + x
        for i in range(1, n):
            current_time = next_departure_time(current_time, bus_info[i - 1][0]) + bus_info[i - 1][1]
        current_time += y
        results.append(current_time)
    
    return results

# Read input
input = sys.stdin.read
data = input().split()
n = int(data[0])
x = int(data[1])
y = int(data[2])
bus_info = [(int(data[3 * i + 3]), int(data[3 * i + 4])) for i in range(n - 1)]
q = int(data[3 * n])
queries = [int(data[3 * n + 1 + i]) for i in range(q)]

# Solve the problem
results = min_time_to_reach_aoki(n, x, y, bus_info, queries)

# Print the results
for result in results:
    print(result)