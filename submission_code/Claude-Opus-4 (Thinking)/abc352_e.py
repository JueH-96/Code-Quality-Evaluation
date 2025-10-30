# YOUR CODE HERE
N, M = map(int, input().split())

# Dictionary to store edges: (u, v) -> weight (where u < v)
edges = {}

for _ in range(M):
    line = list(map(int, input().split()))
    K = line[0]
    C = line[1]
    vertices = line[2:]
    
    # Add edges for all pairs in this subset
    for i in range(K):
        for j in range(i + 1, K):
            u, v = vertices[i], vertices[j]
            if (u, v) not in edges or edges[(u, v)] > C:
                edges[(u, v)] = C

# Build adjacency list
adj = [[] for _ in range(N + 1)]
for (u, v), w in edges.items():
    adj[u].append(v)
    adj[v].append(u)

# DFS to check connectivity
visited = [False] * (N + 1)

def dfs(v):
    visited[v] = True
    for u in adj[v]:
        if not visited[u]:
            dfs(u)

# Start DFS from vertex 1
dfs(1)

# Check if all vertices are visited
connected = all(visited[i] for i in range(1, N + 1))

if not connected:
    print(-1)
else:
    # Find MST using Kruskal's algorithm
    # Convert edges to list and sort by weight
    edge_list = [(w, u, v) for (u, v), w in edges.items()]
    edge_list.sort()
    
    # Union-Find
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1
        return True
    
    mst_weight = 0
    edges_added = 0
    
    for w, u, v in edge_list:
        if union(u, v):
            mst_weight += w
            edges_added += 1
            if edges_added == N - 1:
                break
    
    print(mst_weight)