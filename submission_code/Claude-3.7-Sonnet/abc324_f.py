def has_path_with_ratio_greater_than(N, adj_list, ratio):
    # Initialize distances (highest value for each vertex)
    distances = [-float('inf')] * (N + 1)
    distances[1] = 0
    
    # Process vertices in order (topological sort since u_i < v_i)
    for u in range(1, N + 1):
        for v, beauty, cost in adj_list[u]:
            if distances[u] > -float('inf'):
                distances[v] = max(distances[v], distances[u] + beauty - ratio * cost)
    
    return distances[N] > 0

def solve():
    N, M = map(int, input().split())
    
    # Create an adjacency list
    adj_list = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        u, v, b, c = map(int, input().split())
        adj_list[u].append((v, b, c))  # Store (target, beauty, cost)
    
    # Binary search to find the maximum ratio
    left, right = 0, 10**5  # Upper bound considering max beauty/min cost
    eps = 1e-12  # Precision needed to be within 10^-9
    
    while right - left > eps:
        mid = (left + right) / 2
        
        # Check if there's a path with beauty/cost ratio > mid
        if has_path_with_ratio_greater_than(N, adj_list, mid):
            left = mid
        else:
            right = mid
    
    return left

print(solve())