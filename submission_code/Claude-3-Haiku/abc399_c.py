from collections import defaultdict

def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1

def min_edges_to_delete(N, M, edges):
    parent = [i for i in range(N+1)]
    rank = [0] * (N+1)
    
    # Perform Union-Find to detect cycles
    for u, v in edges:
        if find(parent, u) == find(parent, v):
            # Edge (u, v) forms a cycle, so we need to delete it
            continue
        union(parent, rank, u, v)
    
    # Count the number of edges that need to be deleted
    return M - (N - len(set(find(parent, i) for i in range(1, N+1))))

# Read input
N, M = map(int, input().split())
edges = []
for _ in range(M):
    u, v = map(int, input().split())
    edges.append((u, v))

# Solve the problem
answer = min_edges_to_delete(N, M, edges)
print(answer)