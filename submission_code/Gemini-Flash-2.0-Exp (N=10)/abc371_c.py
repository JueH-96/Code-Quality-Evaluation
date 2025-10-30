import itertools

def solve():
    n = int(input())
    
    mg = int(input())
    g_edges = []
    for _ in range(mg):
        u, v = map(int, input().split())
        g_edges.append((u - 1, v - 1))
    
    mh = int(input())
    h_edges = []
    for _ in range(mh):
        u, v = map(int, input().split())
        h_edges.append((u - 1, v - 1))
    
    costs = []
    for i in range(n - 1):
        line = list(map(int, input().split()))
        costs.append(line)
    
    def get_cost(h_edges_current, target_edges):
        cost = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_exists_h = (i, j) in h_edges_current or (j, i) in h_edges_current
                edge_exists_target = (i, j) in target_edges or (j, i) in target_edges
                if edge_exists_h != edge_exists_target:
                    cost += costs[i][j - i - 1]
        return cost
    
    min_cost = float('inf')
    
    for perm in itertools.permutations(range(n)):
        target_edges = set()
        for u, v in g_edges:
            target_edges.add(tuple(sorted((perm[u], perm[v]))))
        
        current_cost = get_cost(h_edges, target_edges)
        min_cost = min(min_cost, current_cost)
    
    print(min_cost)

solve()