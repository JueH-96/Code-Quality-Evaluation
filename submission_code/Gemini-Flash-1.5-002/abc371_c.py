# YOUR CODE HERE
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
        row = list(map(int, input().split()))
        costs.extend(row)

    cost_matrix = [[0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            cost_matrix[i][j] = costs[k]
            k += 1

    min_cost = float('inf')

    for perm in itertools.permutations(range(n)):
        h_edges_perm = []
        for u, v in h_edges:
            h_edges_perm.append((perm[u], perm[v]))

        cur_cost = 0
        for i in range(n):
            for j in range(i + 1, n):
                g_has_edge = (i, j) in g_edges or (j, i) in g_edges
                h_has_edge = (perm[i], perm[j]) in h_edges_perm or (perm[j], perm[i]) in h_edges_perm
                if g_has_edge != h_has_edge:
                    cur_cost += cost_matrix[i][j]
        min_cost = min(min_cost, cur_cost)

    print(min_cost)

solve()