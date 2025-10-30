import heapq
from itertools import permutations

def dijkstra_minimax(graph, start, n):
    # Find minimum bottleneck paths from start to all vertices
    dist = [float('inf')] * (n + 1)
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            # For minimax path, we take max of current path weight and edge weight
            new_dist = max(d, w)
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    
    return dist

# Read input
n, m, k = map(int, input().split())
graph = [[] for _ in range(n + 1)]

for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))

a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Compute cost matrix: f(a[i], b[j]) for all i, j
cost_matrix = [[0] * k for _ in range(k)]
for i in range(k):
    dist = dijkstra_minimax(graph, a[i], n)
    for j in range(k):
        cost_matrix[i][j] = dist[b[j]]

# Solve assignment problem
# For small k, use brute force
if k <= 8:
    min_cost = float('inf')
    for perm in permutations(range(k)):
        cost = sum(cost_matrix[i][perm[i]] for i in range(k))
        min_cost = min(min_cost, cost)
    print(min_cost)
else:
    # For larger k, use Hungarian algorithm
    # Implementation of Hungarian algorithm
    INF = 10**18
    
    # Arrays for Hungarian algorithm
    u = [0] * (k + 1)  # potential for workers
    v = [0] * (k + 1)  # potential for jobs  
    p = [0] * (k + 1)  # assignment
    way = [0] * (k + 1)
    
    for i in range(1, k + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (k + 1)
        used = [False] * (k + 1)
        
        while p[j0] != 0:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            
            for j in range(1, k + 1):
                if not used[j]:
                    cur = cost_matrix[i0-1][j-1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            
            for j in range(0, k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            
            j0 = j1
        
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    
    # Calculate result
    result = 0
    for j in range(1, k + 1):
        if p[j] != 0:
            result += cost_matrix[p[j]-1][j-1]
    
    print(result)