import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
K = int(data[index + 2])
index += 3

edges = []
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    w = int(data[index + 2])
    edges.append((u, v, w))
    index += 3

A = list(map(int, data[index:index + K]))
B = list(map(int, data[index + K:index + 2 * K]))

# Build the graph
graph = defaultdict(list)
for u, v, w in edges:
    graph[u].append((v, w))
    graph[v].append((u, w))

# Function to find the minimum possible path weight from x to y
def bfs_min_path_weight(x, y):
    queue = deque([(x, 0)])
    visited = set()
    while queue:
        node, max_weight = queue.popleft()
        if node == y:
            return max_weight
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, max(max_weight, weight)))
    return float('inf')

# Precompute f(x, y) for all pairs (A_i, B_j)
f_values = [[0] * K for _ in range(K)]
for i in range(K):
    for j in range(K):
        f_values[i][j] = bfs_min_path_weight(A[i], B[j])

# Hungarian algorithm to find the minimum cost matching
def hungarian_algorithm(cost_matrix):
    n = len(cost_matrix)
    u = [0] * n
    v = [0] * n
    p = [0] * n
    way = [0] * n

    def dfs(v):
        used[v] = True
        for i in range(n):
            if cost_matrix[v][i] - u[v] - v[i] == 0 and not used[i]:
                used[i] = True
                if p[i] == -1 or dfs(p[i]):
                    p[i] = v
                    return True
        return False

    for i in range(n):
        p[i] = -1
        u[i] = cost_matrix[i][0]
        for j in range(1, n):
            u[i] = min(u[i], cost_matrix[i][j])

    for i in range(n):
        while True:
            used = [False] * n
            if dfs(i):
                break
            delta = float('inf')
            for j in range(n):
                if used[j]:
                    for k in range(n):
                        if not used[k]:
                            delta = min(delta, cost_matrix[j][k] - u[j] - v[k])
            for j in range(n):
                if used[j]:
                    u[j] += delta
                if used[j]:
                    v[j] -= delta
                else:
                    way[j] += delta

    return sum(cost_matrix[i][p[i]] for i in range(n))

# Apply Hungarian algorithm to find the minimum sum
min_sum = hungarian_algorithm(f_values)
print(min_sum)