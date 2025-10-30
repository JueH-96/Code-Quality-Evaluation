import itertools

def solve():
    n = int(input())
    m_g = int(input())
    g_edges = []
    for _ in range(m_g):
        u, v = map(int, input().split())
        g_edges.append((u - 1, v - 1))
    
    m_h = int(input())
    h_edges = []
    for _ in range(m_h):
        a, b = map(int, input().split())
        h_edges.append((a - 1, b - 1))
    
    costs = {}
    idx = 0
    for i in range(n):
        if i < n - 1:
            row = list(map(int, input().split()))
            for j in range(i + 1, n):
                costs[(i, j)] = row[j - (i + 1)]
                costs[(j, i)] = row[j - (i + 1)]

    
    g = [[0] * n for _ in range(n)]
    for u, v in g_edges:
        g[u][v] = g[v][u] = 1
    
    h = [[0] * n for _ in range(n)]
    for a, b in h_edges:
        h[a][b] = h[b][a] = 1
        
    
    min_cost = float('inf')
    
    for p in itertools.permutations(range(n)):
        curr_cost = 0
        temp_h = [row[:] for row in h]
        
        for i in range(n):
            for j in range(i + 1, n):
                if g[i][j] == 1 and temp_h[p[i]][p[j]] == 0:
                    curr_cost += costs[(p[i], p[j])]
                    temp_h[p[i]][p[j]] = temp_h[p[j]][p[i]] = 1
                elif g[i][j] == 0 and temp_h[p[i]][p[j]] == 1:
                    curr_cost += costs[(p[i], p[j])]
                    temp_h[p[i]][p[j]] = temp_h[p[j]][p[i]] = 0
        
        
        
        min_cost = min(min_cost, curr_cost)
    
    print(min_cost)

solve()