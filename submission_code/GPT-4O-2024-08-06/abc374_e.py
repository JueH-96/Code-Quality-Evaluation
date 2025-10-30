def can_achieve_capacity(N, X, machines, W):
    total_cost = 0
    for A, P, B, Q in machines:
        # Calculate minimum cost to achieve at least W products for this process
        cost_s = float('inf')
        cost_t = float('inf')
        
        if A > 0:
            s_needed = (W + A - 1) // A  # Ceiling division
            cost_s = s_needed * P
        
        if B > 0:
            t_needed = (W + B - 1) // B  # Ceiling division
            cost_t = t_needed * Q
        
        min_cost = min(cost_s, cost_t)
        total_cost += min_cost
        
        if total_cost > X:
            return False
    
    return total_cost <= X

def max_production_capacity(N, X, machines):
    low, high = 0, 10**9
    best_capacity = 0
    
    while low <= high:
        mid = (low + high) // 2
        if can_achieve_capacity(N, X, machines, mid):
            best_capacity = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_capacity

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
machines = []

index = 2
for _ in range(N):
    A = int(data[index])
    P = int(data[index + 1])
    B = int(data[index + 2])
    Q = int(data[index + 3])
    machines.append((A, P, B, Q))
    index += 4

# Calculate and print the result
result = max_production_capacity(N, X, machines)
print(result)