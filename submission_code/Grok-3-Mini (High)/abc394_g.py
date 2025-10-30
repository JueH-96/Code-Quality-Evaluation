import sys
from collections import deque

# Read all data
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
F = []
for i in range(H):
    row = []
    for j in range(W):
        val = int(data[index])
        row.append(val)
        index += 1
    F.append(row)
N = H * W
floor_heights_node = [0] * N
for i in range(H):
    for j in range(W):
        id_ij = i * W + j
        floor_heights_node[id_ij] = F[i][j]

# Sort cells by floor height decreasing
cells = list(range(N))
cells.sort(key=lambda x: floor_heights_node[x], reverse=True)

# Union-find setup
parent_uf = [0] * N
rank_uf = [0] * N
added = [False] * N
adj_tree = [[] for _ in range(N)]

def find(u):
    while parent_uf[u] != u:
        u = parent_uf[u]
    return u

def union(x, y):
    px = find(x)
    py = find(y)
    if px == py:
        return
    if rank_uf[px] < rank_uf[py]:
        parent_uf[px] = py
    elif rank_uf[px] > rank_uf[py]:
        parent_uf[py] = px
    else:
        parent_uf[py] = px
        rank_uf[px] += 1

# Add nodes in sorted order
for u in cells:
    added[u] = True
    parent_uf[u] = u  # Set as own parent
    row_u = u // W
    col_u = u % W
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni = row_u + di
        nj = col_u + dj
        if 0 <= ni < H and 0 <= nj < W:
            v = ni * W + nj
            if added[v] and find(u) != find(v):
                # Add edge to tree if not already present
                if v not in adj_tree[u]:
                    adj_tree[u].append(v)
                    adj_tree[v].append(u)
                # Union them
                union(u, v)

# Build the tree structure for binary lifting
# Root at node 0
tree_parent = [-1] * N
depth_tree = [0] * N
visited_bfs = [False] * N
queue = deque()
root = 0
visited_bfs[root] = True
depth_tree[root] = 0
tree_parent[root] = -1
queue.append(root)
while queue:
    node = queue.popleft()
    for nei in adj_tree[node]:
        if not visited_bfs[nei]:
            visited_bfs[nei] = True
            tree_parent[nei] = node
            depth_tree[nei] = depth_tree[node] + 1
            queue.append(nei)

# Binary lifting setup
LOGN = 19
up_table = [[-1 for _ in range(LOGN)] for _ in range(N)]
min_f_table = [[0 for _ in range(LOGN)] for _ in range(N)]  # min F on path to ancestor

# Set for k=0
for u in range(N):
    if tree_parent[u] != -1:
        up_table[u][0] = tree_parent[u]
        min_f_table[u][0] = min(floor_heights_node[u], floor_heights_node[tree_parent[u]])
    else:
        up_table[u][0] = -1

# Set for k=1 to LOGN-1
for k_val in range(1, LOGN):
    for u in range(N):
        anc = up_table[u][k_val - 1]
        if anc != -1:
            up_anc = up_table[anc][k_val - 1]
            if up_anc != -1:
                up_table[u][k_val] = up_anc
                min_f_table[u][k_val] = min(min_f_table[u][k_val - 1], min_f_table[anc][k_val - 1])
            else:
                up_table[u][k_val] = -1
        else:
            up_table[u][k_val] = -1

# Define lca function
def lca(p, q):
    if depth_tree[p] > depth_tree[q]:
        p, q = q, p  # swap
    # Lift q up by depth difference
    diff = depth_tree[q] - depth_tree[p]
    for k in range(LOGN):
        if (diff >> k) & 1:
            q = up_table[q][k]
    # Now same depth
    if p == q:
        return p
    for k in range(LOGN - 1, -1, -1):
        if up_table[p][k] != up_table[q][k] and up_table[p][k] != -1:
            p = up_table[p][k]
            q = up_table[q][k]
    return tree_parent[p]

# Define min_to_ancestor function
INF = 2000000000
def min_to_ancestor(u, anc):
    if u == anc:
        return floor_heights_node[u]
    d = depth_tree[u] - depth_tree[anc]
    min_val = INF
    current = u
    for k in range(LOGN):
        if (d >> k) & 1:
            if up_table[current][k] != -1:
                min_val = min(min_val, min_f_table[current][k])
                current = up_table[current][k]
    return min_val

# Read Q
Q = int(data[index])
index += 1

# Process each query
for _ in range(Q):
    S_row = int(data[index]) - 1  # 0-based
    S_col = int(data[index + 1]) - 1
    Y = int(data[index + 2])
    T_row = int(data[index + 3]) - 1
    T_col = int(data[index + 4]) - 1
    Z = int(data[index + 5])
    index += 6
    S_id = S_row * W + S_col
    T_id = T_row * W + T_col
    # Compute X_min using lca and min_to_ancestor
    lca_node = lca(S_id, T_id)
    min_path_S_to_lca = min_to_ancestor(S_id, lca_node)
    min_path_T_to_lca = min_to_ancestor(T_id, lca_node)
    X_min = min(min_path_S_to_lca, min_path_T_to_lca)
    # Compute cost
    M = min(Y, Z)
    if X_min >= M:
        cost = abs(Y - Z)
    else:
        cost = Y + Z - 2 * X_min
    print(cost)