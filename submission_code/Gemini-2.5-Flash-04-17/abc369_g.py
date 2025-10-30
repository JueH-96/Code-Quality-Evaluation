import sys
import heapq

sys.setrecursionlimit(200005)

N = int(sys.stdin.readline())
adj = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    u, v, l = map(int, sys.stdin.readline().split())
    adj[u].append((v, l))
    adj[v].append((u, l))

# DFS for distances, parent, depth, tin/tout
dist = [-1] * (N + 1)
parent = [0] * (N + 1)
depth = [0] * (N + 1)
tin = [-1] * (N + 1)
tout = [-1] * (N + 1)
timer = 0

def dfs_dist(u, p, d, dep):
    global timer
    dist[u] = d
    parent[u] = p
    depth[u] = dep
    tin[u] = timer
    timer += 1
    for v, l in adj[u]:
        if v != p:
            dfs_dist(v, u, d + l, dep + 1)
    tout[u] = timer
    timer += 1

dfs_dist(1, 0, 0, 0)

# Binary lifting for LCA
LOGN = 18
up = [[0] * LOGN for _ in range(N + 1)]
for i in range(1, N + 1):
    up[i][0] = parent[i]

for j in range(1, LOGN):
    for i in range(1, N + 1):
        up[i][j] = up[up[i][j - 1]][j - 1]

def is_ancestor(u, v):
    return tin[u] <= tin[v] and tout[u] >= tout[v]

def lca(u, v):
    if is_ancestor(u, v):
        return u
    if is_ancestor(v, u):
        return v
    for j in range(LOGN - 1, -1, -1):
        if up[u][j] != 0 and not is_ancestor(up[u][j], v):
            u = up[u][j]
    return up[u][0]

def get_dist_lca(u, v):
    lca_node = lca(u, v)
    return dist[lca_node]

max_d_lca_with_selected = [0] * (N + 1)
pq = []
for i in range(1, N + 1):
    # Initial gain for vertex i is d(1,i) - max_d_lca_with_selected[i] = d(1,i) - 0 = d(1,i).
    heapq.heappush(pq, (-dist[i], i)) # Max heap

ans = [0] * (N + 1)
total_gain = 0
selected = [False] * (N + 1)

for k in range(1, N + 1):
    while True:
        current_gain, u = heapq.heappop(pq)
        current_gain = -current_gain
        
        # Check if this is a stale entry
        if selected[u] or (dist[u] - max_d_lca_with_selected[u]) != current_gain:
            continue
        
        # Found the vertex with max gain
        break
    
    total_gain += current_gain
    ans[k] = 2 * total_gain
    selected[u] = True

    # Update gains for all non-selected vertices.
    # This part determines the overall complexity.
    # Only update vertices v where d(1, LCA(v,u)) is potentially > max_d_lca_with_selected[v].
    # This happens if LCA(v,u) is deeper than the current max_d_lca_with_selected[v].
    # Specifically, if LCA(v,u) is on the path 1->u, and this path segment is "newly covered" by u.
    # The set of vertices v whose max_lca_dist changes form specific subtrees.
    # However, checking all remaining vertices and updating if necessary is safer, though slower.
    
    # Collect vertices to update
    vertices_to_update = []
    for v in range(1, N + 1):
        if not selected[v]:
            new_max_d_lca = max(max_d_lca_with_selected[v], get_dist_lca(v, u))
            if new_max_d_lca > max_d_lca_with_selected[v]:
                max_d_lca_with_selected[v] = new_max_d_lca
                vertices_to_update.append(v)

    # Push updated gains to PQ
    for v in vertices_to_update:
        heapq.heappush(pq, (-(dist[v] - max_d_lca_with_selected[v]), v))

for i in range(1, N + 1):
    print(ans[i])