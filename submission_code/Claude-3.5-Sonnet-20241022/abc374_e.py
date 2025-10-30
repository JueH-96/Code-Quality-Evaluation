def can_achieve_capacity(N, X, machines, target_capacity):
    # For each process, find minimum cost to achieve target capacity
    total_cost = 0
    for i in range(N):
        A, P, B, Q = machines[i]
        # Try all combinations of machines S and T
        min_cost = float('inf')
        # Maximum units needed of each type
        max_s = (target_capacity + A - 1) // A if A > 0 else 0
        max_t = (target_capacity + B - 1) // B if B > 0 else 0
        
        # Try each number of S machines
        found = False
        for s in range(max_s + 1):
            s_capacity = s * A
            if s_capacity >= target_capacity:
                min_cost = min(min_cost, s * P)
                found = True
                continue
            # Calculate needed T machines
            remaining = target_capacity - s_capacity
            t = (remaining + B - 1) // B if B > 0 else float('inf')
            if t <= max_t:
                min_cost = min(min_cost, s * P + t * Q)
                found = True
        
        if not found:
            return False
            
        if min_cost > X - total_cost:
            return False
        total_cost += min_cost
    
    return total_cost <= X

def solve():
    # Read input
    N, X = map(int, input().split())
    machines = []
    for _ in range(N):
        A, P, B, Q = map(int, input().split())
        machines.append((A, P, B, Q))
    
    # Binary search for maximum capacity
    left = 0
    right = 10**9 + 1
    
    while right - left > 1:
        mid = (left + right) // 2
        if can_achieve_capacity(N, X, machines, mid):
            left = mid
        else:
            right = mid
    
    print(left)

# Run the solution
solve()