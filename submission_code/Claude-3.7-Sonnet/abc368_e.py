def solve():
    N, M, X_1 = map(int, input().split())
    trains = []
    for _ in range(M):
        A, B, S, T = map(int, input().split())
        trains.append((A, B, S, T))
    
    # Build the graph: edges (i,j) where B_i = A_j and T_i ≤ S_j
    graph = [[] for _ in range(M + 1)]
    for i in range(1, M + 1):
        for j in range(1, M + 1):
            if i != j:  # Avoid self-loops
                if trains[i-1][1] == trains[j-1][0] and trains[i-1][3] <= trains[j-1][2]:
                    weight = trains[i-1][3] - trains[j-1][2]
                    graph[i].append((j, weight))
    
    # Initialize X values
    X = [0] * (M + 1)
    X[1] = X_1
    
    # Bellman-Ford for difference constraints with early stopping
    for _ in range(M - 1):
        updated = False
        for u in range(1, M + 1):
            for v, weight in graph[u]:
                if X[v] < X[u] + weight:
                    X[v] = X[u] + weight
                    updated = True
        if not updated:
            break
    
    return ' '.join(map(str, X[2:]))

print(solve())