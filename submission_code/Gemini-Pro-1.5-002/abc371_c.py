# YOUR CODE HERE
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
        u, v = map(int, input().split())
        h_edges.append((u - 1, v - 1))
    
    a = []
    for _ in range(n - 1):
        a.append(list(map(int, input().split())))
    
    min_cost = float('inf')
    
    for p in itertools.permutations(range(n)):
        cost = 0
        
        transformed_h_edges = []
        for u, v in h_edges:
            transformed_h_edges.append(tuple(sorted((p[u], p[v]))))
        
        transformed_h_edges_set = set(transformed_h_edges)
        
        g_edges_set = set(tuple(sorted(edge)) for edge in g_edges)
        
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in g_edges_set:
                    if (i, j) not in transformed_h_edges_set:
                        cost += a[i][j - i - 1]
                else:
                    if (i, j) in transformed_h_edges_set:
                        cost += a[i][j - i - 1]
        
        min_cost = min(min_cost, cost)
        
    print(min_cost)

solve()