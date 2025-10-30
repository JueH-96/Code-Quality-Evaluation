from itertools import combinations

def min_cost_to_achieve_goal(N, K, P, plans):
    min_cost = float('inf')
    
    # Check all combinations of development plans
    for r in range(1, N + 1):
        for combo in combinations(plans, r):
            total_cost = sum(plan[0] for plan in combo)
            total_params = [0] * K
            
            for plan in combo:
                for j in range(K):
                    total_params[j] += plan[j + 1]
            
            if all(param >= P for param in total_params):
                min_cost = min(min_cost, total_cost)
    
    return min_cost if min_cost != float('inf') else -1

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N, K, P = map(int, data[0].split())
plans = [list(map(int, line.split())) for line in data[1:N + 1]]

# Calculate the result
result = min_cost_to_achieve_goal(N, K, P, plans)

# Print the result
print(result)