def min_cost_to_sort_boxes(N, A, W):
    visited = [False] * N
    min_global_weight = min(W)
    total_cost = 0
    
    for i in range(N):
        if not visited[i]:
            # Start a new cycle
            cycle_weight_sum = 0
            cycle_min_weight = float('inf')
            cycle_length = 0
            x = i
            
            # Traverse the cycle
            while not visited[x]:
                visited[x] = True
                cycle_weight_sum += W[x]
                cycle_min_weight = min(cycle_min_weight, W[x])
                cycle_length += 1
                x = A[x] - 1  # Move to the next item in the cycle
            
            if cycle_length > 1:
                # Calculate the cost of resolving this cycle
                # Method 1: Using the cycle's minimum weight
                cost1 = cycle_weight_sum + (cycle_length - 2) * cycle_min_weight
                # Method 2: Using the global minimum weight
                cost2 = cycle_weight_sum + cycle_min_weight + (cycle_length + 1) * min_global_weight
                # Add the minimum of the two costs to the total cost
                total_cost += min(cost1, cost2)
    
    return total_cost

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
W = list(map(int, data[N+1:2*N+1]))

# Calculating the minimum cost
result = min_cost_to_sort_boxes(N, A, W)

# Output the result
print(result)