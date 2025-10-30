import sys
import heapq
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
K = int(data[2])

edges = []
for i in range(M):
    u = int(data[3 + 3 * i])
    v = int(data[4 + 3 * i])
    w = int(data[5 + 3 * i])
    edges.append((u, v, w))

A = [int(data[3 + 3 * M + i]) for i in range(K)]
B = [int(data[3 + 3 * M + K + i]) for i in range(K)]

# Create an adjacency list
graph = defaultdict(list)
for u, v, w in edges:
    graph[u].append((v, w))
    graph[v].append((u, w))

# Function to find the minimum possible path weight using Dijkstra's algorithm
def find_min_path_weight(start, end):
    min_heap = [(0, start)]
    visited = set()
    max_weight = [float('inf')] * (N + 1)
    max_weight[start] = 0

    while min_heap:
        current_weight, current_node = heapq.heappop(min_heap)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == end:
            return max_weight[end]

        for neighbor, weight in graph[current_node]:
            if neighbor not in visited:
                new_weight = max(current_weight, weight)
                if new_weight < max_weight[neighbor]:
                    max_weight[neighbor] = new_weight
                    heapq.heappush(min_heap, (new_weight, neighbor))

    return max_weight[end]

# Calculate the minimum possible path weight for all pairs (A_i, B_j)
cost_matrix = [[0] * K for _ in range(K)]
for i in range(K):
    for j in range(K):
        cost_matrix[i][j] = find_min_path_weight(A[i], B[j])

# Use the Hungarian algorithm to find the minimum cost matching
def hungarian_algorithm(cost_matrix):
    n = len(cost_matrix)
    u = [0] * n
    v = [0] * n
    p = [0] * n
    way = [0] * n

    for i in range(n):
        links = [False] * n
        minv = [float('inf')] * n
        links[i] = True
        j = 0
        t = 0
        while True:
            j = way[j]
            links[j] = True
            delta = float('inf')
            t = p[j]
            for k in range(n):
                if not links[k]:
                    cur = cost_matrix[j][k] - u[j] - v[k]
                    if cur < minv[k]:
                        minv[k] = cur
                        way[k] = j
                    if minv[k] < delta:
                        delta = minv[k]
                        t = k
            for k in range(n):
                if links[k]:
                    u[k] += delta
                    v[t] -= delta
                else:
                    minv[k] -= delta
            links[t] = True
            if p[t] == -1:
                break

    return -v[0]

# Calculate the minimum cost
min_cost = hungarian_algorithm(cost_matrix)

# Output the result
sys.stdout.write(str(min_cost) + '
')