def find(u):
    if parent[u] != u:
        parent[u] = find(parent[u])
    return parent[u]

def union(u, v):
    u_root = find(u)
    v_root = find(v)
    if u_root == v_root:
        return False
    if rank[u_root] < rank[v_root]:
        parent[u_root] = v_root
    else:
        parent[v_root] = u_root
        if rank[u_root] == rank[v_root]:
            rank[u_root] += 1
    return True

def dfs(current_sum, num_edges, edge_index):
    nonlocal min_cost
    if num_edges == N - 1:
        min_cost = min(min_cost, current_sum % K)
        return
    for i in range(edge_index, M):
        u, v, w = edges[i]
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent_copy = parent.copy()
            rank_copy = rank.copy()
            union(u, v)
            dfs(current_sum + w, num_edges + 1, i + 1)
            parent = parent_copy
            rank = rank_copy
        else:
            continue

# Read input
N, M, K = map(int, input().split())
edges = []
for _ in range(M):
    u, v, w = map(int, input().split())
    u -= 1  # Convert to 0-indexed
    v -= 1  # Convert to 0-indexed
    edges.append((u, v, w))

# Initialize union-find
parent = list(range(N))
rank = [1] * N

# Initialize min_cost
min_cost = K - 1

# Start DFS
dfs(0, 0, 0)

# Print the answer
print(min_cost)