import itertools

n = int(input())

# Read graph G
m_g = int(input())
g_edges = set()
for _ in range(m_g):
    u, v = map(int, input().split())
    if u > v:
        u, v = v, u
    g_edges.add((u, v))

# Read graph H
m_h = int(input())
h_edges = set()
for _ in range(m_h):
    a, b = map(int, input().split())
    if a > b:
        a, b = b, a
    h_edges.add((a, b))

# Read cost matrix A
a = [[0] * (n + 1) for _ in range(n + 1)]  # a[i][j] for i < j
for i in range(1, n):
    row = list(map(int, input().split()))
    for j in range(i + 1, n + 1):
        a[i][j] = row[j - (i + 1)]

min_cost = float('inf')

# Check all permutations of node mappings
for perm in itertools.permutations(range(1, n + 1)):
    current_cost = 0
    for i in range(n):
        for j in range(i + 1, n):
            u = i + 1
            v = j + 1
            a_node = perm[i]
            b_node = perm[j]
            sorted_edge = tuple(sorted((a_node, b_node)))
            if sorted_edge in g_edges:
                if (u, v) not in h_edges:
                    current_cost += a[u][v]
            else:
                if (u, v) in h_edges:
                    current_cost += a[u][v]
    if current_cost < min_cost:
        min_cost = current_cost

print(min_cost)