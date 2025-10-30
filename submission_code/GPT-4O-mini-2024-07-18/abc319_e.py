def earliest_arrival(N, X, Y, P, T, Q, queries):
    results = []
    
    for q in queries:
        # Start time is when Takahashi leaves his house
        current_time = q + X  # Time when he arrives at bus stop 1
        
        for i in range(N - 1):
            # Calculate the next bus departure time
            if current_time % P[i] == 0:
                bus_departure_time = current_time
            else:
                bus_departure_time = (current_time // P[i] + 1) * P[i]
            
            # Travel to the next bus stop
            current_time = bus_departure_time + T[i]
        
        # Finally walk to Aoki's house
        current_time += Y
        
        results.append(current_time)
    
    return results

import sys
input = sys.stdin.read
data = input().splitlines()

# Read the first line
N, X, Y = map(int, data[0].split())
P = []
T = []

# Read the bus stop information
for i in range(1, N):
    p_i, t_i = map(int, data[i].split())
    P.append(p_i)
    T.append(t_i)

# Read the number of queries
Q = int(data[N])
queries = [int(data[N + 1 + i]) for i in range(Q)]

# Get the results
results = earliest_arrival(N, X, Y, P, T, Q, queries)

# Print the results
for result in results:
    print(result)