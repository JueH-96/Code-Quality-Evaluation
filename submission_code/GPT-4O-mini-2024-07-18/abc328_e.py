from itertools import combinations

def find_minimum_spanning_tree_cost(N, M, K, edges):
    min_cost = float('inf')
    
    # Generate all combinations of N-1 edges from the M edges
    for edge_combination in combinations(edges, N - 1):
        # Create a set to track connected components
        parent = list(range(N + 1))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX
        
        total_weight = 0
        valid_tree = True
        
        for u, v, w in edge_combination:
            if find(u) != find(v):
                union(u, v)
                total_weight += w
            else:
                valid_tree = False
                break
        
        if valid_tree:
            min_cost = min(min_cost, total_weight % K)
    
    return min_cost

import sys
input = sys.stdin.read
data = input().splitlines()

# Read N, M, K
N, M, K = map(int, data[0].split())
edges = []

# Read edges
for i in range(1, M + 1):
    u, v, w = map(int, data[i].split())
    edges.append((u, v, w))

# Calculate the minimum cost of a spanning tree
result = find_minimum_spanning_tree_cost(N, M, K, edges)

# Print the result
print(result)