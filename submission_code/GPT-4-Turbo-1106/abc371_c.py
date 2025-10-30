from itertools import permutations

def read_graph(M):
    edges = set()
    for _ in range(M):
        u, v = map(int, input().split())
        edges.add(frozenset((u, v)))
    return edges

def calculate_cost(G, H, cost_matrix, perm):
    cost = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (frozenset((i + 1, j + 1)) in G) != (frozenset((perm[i], perm[j])) in H):
                cost += cost_matrix[i][j]
    return cost

N = int(input())
M_G = int(input())
G = read_graph(M_G)
M_H = int(input())
H = read_graph(M_H)

cost_matrix = [[0] * N for _ in range(N)]
for i in range(N - 1):
    row = list(map(int, input().split()))
    for j, cost in enumerate(row, start=i + 1):
        cost_matrix[i][j] = cost

min_cost = float('inf')
for perm in permutations(range(1, N + 1)):
    min_cost = min(min_cost, calculate_cost(G, H, cost_matrix, perm))

print(min_cost)