def min_cost_to_monitor_sections(N, D, L1, C1, K1, L2, C2, K2):
    # Calculate total lengths needed to monitor
    total_length = sum(D)
    
    # Calculate maximum coverage with each type of sensor
    max_coverage1 = L1 * K1
    max_coverage2 = L2 * K2
    
    # If the total length is greater than the combined maximum coverage, it's impossible
    if total_length > max_coverage1 + max_coverage2:
        return -1
    
    # Initialize the minimum cost to a large number
    min_cost = float('inf')
    
    # Try all possible counts of type-1 sensors from 0 to K1
    for count1 in range(K1 + 1):
        # Calculate the length covered by type-1 sensors
        covered_length1 = count1 * L1
        
        # Remaining length to cover
        remaining_length = total_length - covered_length1
        
        # If remaining length is less than or equal to 0, we can cover everything with type-1 sensors
        if remaining_length <= 0:
            cost = count1 * C1
            min_cost = min(min_cost, cost)
            continue
        
        # Calculate how many type-2 sensors are needed
        count2 = (remaining_length + L2 - 1) // L2  # Ceiling division
        
        # Check if we can use this many type-2 sensors
        if count2 <= K2:
            cost = count1 * C1 + count2 * C2
            min_cost = min(min_cost, cost)
    
    return min_cost if min_cost != float('inf') else -1

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = list(map(int, data[1:N+1]))
L1, C1, K1 = map(int, data[N+1:N+4])
L2, C2, K2 = map(int, data[N+4:N+7])

result = min_cost_to_monitor_sections(N, D, L1, C1, K1, L2, C2, K2)
print(result)