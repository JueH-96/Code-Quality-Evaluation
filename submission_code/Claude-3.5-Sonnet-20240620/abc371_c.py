# YOUR CODE HERE
import sys
from itertools import permutations

def read_graph(m):
    return set(tuple(sorted(map(int, input().split()))) for _ in range(m))

def read_costs(n):
    costs = {}
    for i in range(1, n):
        row = list(map(int, input().split()))
        for j, cost in enumerate(row, i + 1):
            costs[i, j] = cost
    return costs

def graph_to_adj(graph, n):
    adj = [[0] * n for _ in range(n)]
    for u, v in graph:
        adj[u-1][v-1] = adj[v-1][u-1] = 1
    return adj

def calculate_cost(g, h, perm, costs):
    total_cost = 0
    n = len(g)
    for i in range(n):
        for j in range(i+1, n):
            if g[i][j] != h[perm[i]][perm[j]]:
                total_cost += costs[min(perm[i]+1, perm[j]+1), max(perm[i]+1, perm[j]+1)]
    return total_cost

def solve(n, g, h, costs):
    g_adj = graph_to_adj(g, n)
    h_adj = graph_to_adj(h, n)
    
    min_cost = float('inf')
    for perm in permutations(range(n)):
        cost = calculate_cost(g_adj, h_adj, perm, costs)
        min_cost = min(min_cost, cost)
    
    return min_cost

n = int(input())
m_g = int(input())
g = read_graph(m_g)
m_h = int(input())
h = read_graph(m_h)
costs = read_costs(n)

result = solve(n, g, h, costs)
print(result)