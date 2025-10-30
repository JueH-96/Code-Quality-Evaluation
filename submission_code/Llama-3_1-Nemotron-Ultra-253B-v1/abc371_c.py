import itertools

n = int(input())

m_g = int(input())
g_edges = set()
for _ in range(m_g):
    u, v = map(int, input().split())
    if u > v:
        u, v = v, u
    g_edges.add((u, v))

m_h = int(input())
h_edges = set()
for _ in range(m_h):
    a, b = map(int, input().split())
    if a > b:
        a, b = b, a
    h_edges.add((a, b))

A = [[0] * (n + 1) for _ in range(n + 1)]
for i in range(1, n):
    values = list(map(int, input().split()))
    for idx, j in enumerate(range(i + 1, n + 1)):
        A[i][j] = values[idx]

min_cost = float('inf')

for perm in itertools.permutations(range(1, n + 1)):
    t_p = set()
    for u, v in g_edges:
        pu = perm[u - 1]
        pv = perm[v - 1]
        a = min(pu, pv)
        b = max(pu, pv)
        t_p.add((a, b))
    
    cost = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i, j) in t_p:
                if (i, j) not in h_edges:
                    cost += A[i][j]
            else:
                if (i, j) in h_edges:
                    cost += A[i][j]
    
    if cost < min_cost:
        min_cost = cost

print(min_cost)