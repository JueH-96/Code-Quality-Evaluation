import sys
import itertools

def solve():
    n = int(sys.stdin.readline())
    m_g = int(sys.stdin.readline())
    edges_g = []
    for _ in range(m_g):
        u, v = map(int, sys.stdin.readline().split())
        edges_g.append((u, v))
    m_h = int(sys.stdin.readline())
    edges_h = []
    for _ in range(m_h):
        a, b = map(int, sys.stdin.readline().split())
        edges_h.append((a, b))
    costs_input = []
    for _ in range(n - 1):
        costs_input.append(list(map(int, sys.stdin.readline().split())))
    
    costs = {}
    cost_index = 0
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            costs[(i, j)] = costs_input[i-1][j-i-1]
            
    adj_g = [[0] * n for _ in range(n)]
    for u, v in edges_g:
        adj_g[u-1][v-1] = adj_g[v-1][u-1] = 1
        
    adj_h = [[0] * n for _ in range(n)]
    for a, b in edges_h:
        adj_h[a-1][b-1] = adj_h[b-1][a-1] = 1
        
    min_cost = float('inf')
    
    permutations = list(itertools.permutations(range(n)))
    
    for p in permutations:
        current_cost = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_in_g = adj_g[i][j]
                edge_in_h_permuted = adj_h[p[i]][p[j]]
                if edge_in_g != edge_in_h_permuted:
                    u_h, v_h = sorted((p[i] + 1, p[j] + 1))
                    current_cost += costs[(u_h, v_h)]
                    
        min_cost = min(min_cost, current_cost)
        
    print(min_cost)

if __name__ == '__main__':
    solve()