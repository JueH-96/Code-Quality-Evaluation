import sys
from collections import defaultdict, deque

def solve():
    N, M, X1 = map(int, input().split())
    trains = []
    for i in range(M):
        A, B, S, T = map(int, input().split())
        trains.append((A, B, S, T))
    
    # Build constraint graph
    # For constraint X_j - X_i >= c, add edge i -> j with weight c
    graph = defaultdict(list)
    
    # Add constraints based on transfer conditions
    for i in range(M):
        for j in range(M):
            if i != j:
                A_i, B_i, S_i, T_i = trains[i]
                A_j, B_j, S_j, T_j = trains[j]
                
                # If train i arrives at city where train j departs
                # and originally transfer was possible
                if B_i == A_j and T_i <= S_j:
                    # Constraint: T_i + X_i <= S_j + X_j
                    # Rearranged: X_j - X_i >= T_i - S_j
                    constraint_value = T_i - S_j
                    graph[i].append((j, constraint_value))
    
    # Use Bellman-Ford algorithm to find minimum values
    # Initialize distances
    dist = [float('inf')] * M
    dist[0] = X1  # X_1 is given
    
    # Relax edges M-1 times
    for _ in range(M - 1):
        updated = False
        for u in range(M):
            if dist[u] != float('inf'):
                for v, weight in graph[u]:
                    if dist[u] + weight > dist[v]:
                        dist[v] = dist[u] + weight
                        updated = True
        if not updated:
            break
    
    # Handle nodes not reachable from node 0
    # Use BFS to find all reachable nodes
    visited = [False] * M
    queue = deque([0])
    visited[0] = True
    
    while queue:
        u = queue.popleft()
        for v, _ in graph[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    
    # For unreachable nodes, set minimum value (0)
    for i in range(M):
        if not visited[i]:
            dist[i] = 0
    
    # Output X_2, ..., X_M
    result = []
    for i in range(1, M):
        result.append(str(dist[i]))
    
    print(' '.join(result))

solve()