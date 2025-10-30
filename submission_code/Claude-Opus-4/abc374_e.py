def min_cost_for_capacity(target, A, P, B, Q):
    """
    Find minimum cost to achieve at least 'target' capacity for a single process
    using machines with (capacity A, cost P) and (capacity B, cost Q)
    """
    if target == 0:
        return 0
    
    min_cost = float('inf')
    
    # Try different combinations of machine S
    max_s = (target + A - 1) // A  # Ceiling division
    for s_count in range(max_s + 1):
        remaining = max(0, target - s_count * A)
        if remaining > 0:
            t_count = (remaining + B - 1) // B  # Ceiling division
        else:
            t_count = 0
        cost = s_count * P + t_count * Q
        min_cost = min(min_cost, cost)
    
    # Also try using only machine T
    t_only = (target + B - 1) // B
    min_cost = min(min_cost, t_only * Q)
    
    return min_cost

def can_achieve_capacity(capacity, processes, budget):
    """
    Check if we can achieve the given capacity within budget
    """
    total_cost = 0
    for A, P, B, Q in processes:
        cost = min_cost_for_capacity(capacity, A, P, B, Q)
        if cost == float('inf'):
            return False
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

# Binary search on the production capacity
left, right = 0, 10**9 + 1

while right - left > 1:
    mid = (left + right) // 2
    if can_achieve_capacity(mid, processes, X):
        left = mid
    else:
        right = mid

print(left)