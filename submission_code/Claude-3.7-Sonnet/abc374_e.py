def min_cost_to_achieve_capacity(capacity, a, p, b, q):
    """
    Find the minimum cost to achieve a given capacity for a single process.
    """
    if capacity == 0:
        return 0
    
    if a == 0 and b == 0:
        return float('inf')  # Can't achieve any positive capacity
    
    if a == 0:
        t = (capacity + b - 1) // b  # Ceiling division
        return t * q
    
    if b == 0:
        s = (capacity + a - 1) // a  # Ceiling division
        return s * p
    
    min_cost = float('inf')
    max_s = (capacity + a - 1) // a  # Ceiling division
    
    for s in range(max_s + 1):
        s_capacity = s * a
        remaining = max(0, capacity - s_capacity)
        t = (remaining + b - 1) // b if remaining > 0 else 0  # Ceiling division
        
        cost = s * p + t * q
        min_cost = min(min_cost, cost)
    
    return min_cost

def is_achievable(capacity, A, P, B, Q, X):
    """
    Check if the given capacity is achievable within the budget X.
    """
    total_cost = 0
    for a, p, b, q in zip(A, P, B, Q):
        cost = min_cost_to_achieve_capacity(capacity, a, p, b, q)
        if cost == float('inf'):
            return False
        total_cost += cost
        if total_cost > X:
            return False
    
    return True

def max_production_capacity(N, X, A, P, B, Q):
    """
    Find the maximum achievable production capacity within the budget X.
    """
    # Binary search to find the maximum capacity
    left, right = 0, 10**9 + 1  # Upper bound based on constraints
    
    while left <= right:
        mid = (left + right) // 2
        if is_achievable(mid, A, P, B, Q, X):
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Right is the maximum achievable capacity

# Read input
N, X = map(int, input().split())
A, P, B, Q = [], [], [], []

for _ in range(N):
    a, p, b, q = map(int, input().split())
    A.append(a)
    P.append(p)
    B.append(b)
    Q.append(q)

# Compute and print the maximum production capacity
print(max_production_capacity(N, X, A, P, B, Q))