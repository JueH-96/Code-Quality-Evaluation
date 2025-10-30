import math

def min_cost_for_capacity(A, P, B, Q, target):
    if target == 0:
        return 0
    
    min_cost = float('inf')
    
    # Try different numbers of S machines
    max_s = (target + A - 1) // A  # Maximum S machines needed
    
    for s in range(max_s + 1):
        remaining = max(0, target - s * A)
        if remaining == 0:
            t = 0
        else:
            t = (remaining + B - 1) // B  # Ceiling division
        
        cost = s * P + t * Q
        min_cost = min(min_cost, cost)
    
    return min_cost

def can_achieve_capacity(processes, budget, target):
    total_cost = 0
    for A, P, B, Q in processes:
        cost = min_cost_for_capacity(A, P, B, Q, target)
        total_cost += cost
        if total_cost > budget:
            return False
    return True

# Read input
N, X = map(int, input().split())
processes = []
for _ in range(N):
    A, P, B, Q = map(int, input().split())
    processes.append((A, P, B, Q))

# Binary search on the answer
left, right = 0, 10**9

while left < right:
    mid = (left + right + 1) // 2
    if can_achieve_capacity(processes, X, mid):
        left = mid
    else:
        right = mid - 1

print(left)