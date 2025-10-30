from collections import defaultdict
from itertools import permutations

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    px, py = find(parent, x), find(parent, y)
    if px == py:
        return False
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1
    return True

def kruskal(n, edges):
    edges.sort()
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    mst_edges = []
    
    for w, u, v in edges:
        if union(parent, rank, u, v):
            mst_edges.append((u, v, w))
    
    return mst_edges

def build_tree(n, mst_edges):
    tree = defaultdict(list)
    for u, v, w in mst_edges:
        tree[u].append((v, w))
        tree[v].append((u, w))
    return tree

def find_max_edge_on_path(tree, start, end):
    if start == end:
        return 0
    
    def dfs(node, parent, target, max_edge):
        if node == target:
            return max_edge
        
        for neighbor, edge_weight in tree[node]:
            if neighbor != parent:
                result = dfs(neighbor, node, target, max(max_edge, edge_weight))
                if result is not None:
                    return result
        
        return None
    
    return dfs(start, -1, end, 0)

n, m, k = map(int, input().split())

edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((w, u, v))

a = list(map(int, input().split()))
b = list(map(int, input().split()))

mst_edges = kruskal(n, edges)
tree = build_tree(n, mst_edges)

cost_matrix = [[0] * k for _ in range(k)]
for i in range(k):
    for j in range(k):
        cost_matrix[i][j] = find_max_edge_on_path(tree, a[i], b[j])

min_cost = float('inf')
for perm in permutations(range(k)):
    cost = sum(cost_matrix[i][perm[i]] for i in range(k))
    min_cost = min(min_cost, cost)

print(min_cost)