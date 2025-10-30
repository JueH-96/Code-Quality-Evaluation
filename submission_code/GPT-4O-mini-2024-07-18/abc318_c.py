def minimum_cost_trip(N, D, P, fares):
    total_cost = 0
    # Calculate the total cost if we were to pay for each day
    total_regular_fare = sum(fares)
    
    # Calculate the number of passes needed
    passes_needed = (N + D - 1) // D  # This is equivalent to ceil(N / D)
    
    # Calculate the cost if we buy passes for all days
    cost_with_passes = passes_needed * P
    
    # The minimum cost will be the lesser of using passes for all days or paying regular fare
    min_cost = min(total_regular_fare, cost_with_passes)
    
    # Now we need to consider mixed strategies
    # We will iterate through the fares and calculate the cost dynamically
    current_cost = 0
    for i in range(N):
        current_cost += fares[i]
        # If we have covered enough days to buy passes
        if (i + 1) % D == 0:
            # Calculate the cost if we buy passes for the last D days
            cost_with_current_passes = (i + 1) // D * P + (total_regular_fare - current_cost)
            min_cost = min(min_cost, cost_with_current_passes)
    
    return min_cost

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
D = int(data[1])
P = int(data[2])
fares = list(map(int, data[3:]))

result = minimum_cost_trip(N, D, P, fares)
print(result)