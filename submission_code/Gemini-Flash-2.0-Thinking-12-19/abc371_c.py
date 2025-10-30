import sys
import itertools

def solve():
    n = int(sys.stdin.readline())
    m_g = int(sys.stdin.readline())
    g_edges = []
    for _ in range(m_g):
        u, v = map(int, sys.stdin.readline().split())
        g_edges.append((u, v))
    m_h = int(sys.stdin.readline())
    h_edges = []
    for _ in range(m_h):
        a, b = map(int, sys.stdin.readline().split())
        h_edges.append((a, b))
    costs_input = []
    for _ in range(n - 1):
        costs_input.extend(list(map(int, sys.stdin.readline().split())))
    
    costs = {}
    cost_index = 0
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            costs[(i, j)] = costs_input[cost_index]
            cost_index += 1
            
    g_adj = [[0] * (n + 1) for _ in range(n + 1)]
    for u, v in g_edges:
        g_adj[u][v] = g_adj[v][u] = 1
        
    h_adj = [[0] * (n + 1) for _ in range(n + 1)]
    for u, v in h_edges:
        h_adj[u][v] = h_adj[v][u] = 1
        
    min_cost = float('inf')
    
    permutations = list(itertools.permutations(range(1, n + 1)))
    
    for p in permutations:
        current_cost = 0
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                u, v = i, j
                pu, pv = p[i-1], p[j-1]
                g_edge = g_adj[u][v]
                h_edge = h_adj[pu][pv]
                if g_edge == 1 and h_edge == 0:
                    current_cost += costs[(min(pu, pv), max(pu, pv))]
                elif g_edge == 0 and h_edge == 1:
                    current_cost += costs[(min(pu, pv), max(pu, pv))]
                    
        min_cost = min(min_cost, current_cost)
        
    print(min_cost)

if __name__ == '__main__':
    solve()