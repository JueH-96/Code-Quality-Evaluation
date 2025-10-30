def min_cost_for_capacity(A, P, B, Q, C):
    if C == 0:
        return 0
    
    min_cost = float('inf')
    
    # Enumerate over x (number of S machines)
    max_x = min((C + A - 1) // A, 10**6)  # Limit to avoid TLE
    for x in range(max_x + 1):
        remaining = C - x * A
        if remaining <= 0:
            y = 0
        else:
            y = (remaining + B - 1) // B  # ceil(remaining / B)
        
        cost = x * P + y * Q
        min_cost = min(min_cost, cost)
    
    return min_cost

def can_achieve_capacity(processes, X, C):
    total_cost = 0
    for A, P, B, Q in processes:
        cost = min_cost_for_capacity(A, P, B, Q, C)
        total_cost += cost
        if total_cost > X:
            return False
    return True

N, X = map(int, input().split())
processes = []
for _ in range(N):
    A, P, B, Q = map(int, input().split())
    processes.append((A, P, B, Q))

# Compute an upper bound for the binary search
upper_bound = float('inf')
for A, P, B, Q in processes:
    max_capacity_S = X // P * A if P <= X else 0
    max_capacity_T = X // Q * B if Q <= X else 0
    max_capacity = max(max_capacity_S, max_capacity_T)
    upper_bound = min(upper_bound, max_capacity)

if upper_bound == float('inf'):
    upper_bound = 0

# Binary search on the capacity
left, right = 0, int(upper_bound)

while left < right:
    mid = (left + right + 1) // 2
    if can_achieve_capacity(processes, X, mid):
        left = mid
    else:
        right = mid - 1

print(left)