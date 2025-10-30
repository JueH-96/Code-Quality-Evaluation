from itertools import permutations

def get_edges(n, m, edges):
    # Create adjacency matrix
    adj = [[0]*n for _ in range(n)]
    for u,v in edges:
        adj[u-1][v-1] = adj[v-1][u-1] = 1
    return adj

def get_cost_matrix(n, costs):
    # Create cost matrix from upper triangular costs
    cost = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            cost[i][j] = cost[j][i] = costs[idx]
            idx += 1
    return cost

def calc_cost(g, h, cost_matrix, perm):
    # Calculate cost to transform h to match g under permutation
    n = len(g)
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            pi, pj = perm[i], perm[j]
            if g[i][j] != h[pi][pj]:
                total += cost_matrix[pi][pj]
    return total

def solve():
    # Read input
    n = int(input())
    mg = int(input())
    g_edges = []
    for _ in range(mg):
        u,v = map(int, input().split())
        g_edges.append((u,v))
        
    mh = int(input())
    h_edges = []
    for _ in range(mh):
        u,v = map(int, input().split())
        h_edges.append((u,v))
        
    costs = []
    for i in range(n-1):
        costs.extend(map(int, input().split()))
        
    # Create adjacency matrices
    g = get_edges(n, mg, g_edges)
    h = get_edges(n, mh, h_edges)
    
    # Create cost matrix
    cost_matrix = get_cost_matrix(n, costs)
    
    # Try all permutations
    min_cost = float('inf')
    for perm in permutations(range(n)):
        cost = calc_cost(g, h, cost_matrix, perm)
        min_cost = min(min_cost, cost)
        
    print(min_cost)

solve()