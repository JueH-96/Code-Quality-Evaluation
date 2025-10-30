# YOUR CODE HERE
def min_cost_for_process(A, P, B, Q, C):
    # Find minimum cost to achieve at least C products for this process
    min_cost = float('inf')
    
    # Try using only S machines
    if A > 0:
        x = (C + A - 1) // A  # ceiling division
        cost = x * P
        min_cost = min(min_cost, cost)
    
    # Try using only T machines
    if B > 0:
        y = (C + B - 1) // B  # ceiling division
        cost = y * Q
        min_cost = min(min_cost, cost)
    
    # Try mixed strategies - iterate through different values of x
    # Limit iterations for large C to avoid TLE
    if A > 0:
        max_x = min((C + A - 1) // A, 100000)
        for x in range(max_x + 1):
            remaining = C - x * A
            if remaining <= 0:
                cost = x * P
            else:
                if B > 0:
                    y = (remaining + B - 1) // B
                    cost = x * P + y * Q
                else:
                    continue
            min_cost = min(min_cost, cost)
    
    return min_cost

def can_achieve_capacity(processes, X, C):
    # Check if we can achieve production capacity C within budget X
    total_cost = 0
    for A, P, B, Q in processes:
        cost = min_cost_for_process(A, P, B, Q, C)
        if cost == float('inf'):
            return False
        total_cost += cost
        if total_cost > X:
            return False
    return True

# Read input
N, X = map(int, input().split())
processes = []
for _ in range(N):
    A, P, B, Q = map(int, input().split())
    processes.append((A, P, B, Q))

# Binary search on the production capacity
left, right = 0, 10**9
while left < right:
    mid = (left + right + 1) // 2
    if can_achieve_capacity(processes, X, mid):
        left = mid
    else:
        right = mid - 1

print(left)