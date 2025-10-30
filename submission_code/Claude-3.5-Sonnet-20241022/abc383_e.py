def floyd_warshall(n, edges):
    # Initialize distance matrix with infinity
    dist = [[float('inf')] * (n+1) for _ in range(n+1)]
    
    # Set diagonal to 0
    for i in range(1, n+1):
        dist[i][i] = 0
    
    # Initialize with direct edges
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w
    
    # Floyd-Warshall algorithm for min-max path
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                dist[i][j] = min(dist[i][j], 
                               max(dist[i][k], dist[k][j]))
    
    return dist

def hungarian_algorithm(cost_matrix):
    n = len(cost_matrix)
    u = [0] * n
    v = [0] * n
    match = [-1] * n
    
    for i in range(n):
        links = [-1] * n
        mins = [float('inf')] * n
        visited = [False] * n
        markedI = i
        markedJ = -1
        while markedI != -1:
            j = -1
            for j1 in range(n):
                if not visited[j1]:
                    cur = cost_matrix[markedI][j1] - u[markedI] - v[j1]
                    if cur < mins[j1]:
                        mins[j1] = cur
                        links[j1] = markedJ
                    if j == -1 or mins[j1] < mins[j]:
                        j = j1
            delta = mins[j]
            for j1 in range(n):
                if visited[j1]:
                    u[match[j1]] += delta
                    v[j1] -= delta
                else:
                    mins[j1] -= delta
            u[i] += delta
            visited[j] = True
            markedJ = j
            markedI = match[j]
        while links[j] != -1:
            match[j] = match[links[j]]
            j = links[j]
        match[j] = i
    return match

def solve():
    # Read input
    N, M, K = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Get all shortest paths using Floyd-Warshall
    dist = floyd_warshall(N, edges)
    
    # Create cost matrix for Hungarian algorithm
    cost_matrix = []
    for i in range(K):
        row = []
        for j in range(K):
            row.append(dist[A[i]][B[j]])
        cost_matrix.append(row)
    
    # Get optimal assignment
    matching = hungarian_algorithm(cost_matrix)
    
    # Calculate total cost
    total_cost = sum(cost_matrix[i][matching[i]] for i in range(K))
    
    print(total_cost)

solve()