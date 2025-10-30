from collections import defaultdict

def is_isomorphic(g, h):
    if len(g) != len(h):
        return False
    
    g_to_h = {}
    h_to_g = {}
    
    for i in range(len(g)):
        u, v = g[i]
        a, b = h[i]
        
        if (u, v) in g_to_h:
            if g_to_h[(u, v)] != (a, b):
                return False
        else:
            g_to_h[(u, v)] = (a, b)
        
        if (a, b) in h_to_g:
            if h_to_g[(a, b)] != (u, v):
                return False
        else:
            h_to_g[(a, b)] = (u, v)
    
    return True

def solve(n, m_g, g, m_h, h, costs):
    min_cost = float('inf')
    
    for mask in range(1 << (n * (n - 1) // 2)):
        h_new = [edge for edge in h]
        cost = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if mask & (1 << ((i * (n - 1)) + j - 1)):
                    if (i + 1, j + 1) in h_new:
                        h_new.remove((i + 1, j + 1))
                        cost += costs[i][j - 1]
                    else:
                        h_new.append((i + 1, j + 1))
                        cost += costs[i][j - 1]
        
        if is_isomorphic(g, h_new):
            min_cost = min(min_cost, cost)
    
    return min_cost

# Read input
n = int(input())
m_g = int(input())
g = []
for _ in range(m_g):
    u, v = map(int, input().split())
    g.append((u, v))
m_h = int(input())
h = []
for _ in range(m_h):
    a, b = map(int, input().split())
    h.append((a, b))
costs = []
for i in range(n):
    row = list(map(int, input().split()))
    costs.append(row[i:])

# Solve the problem
print(solve(n, m_g, g, m_h, h, costs))