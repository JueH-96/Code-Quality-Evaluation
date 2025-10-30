import sys

sys.setrecursionlimit(300000)

N = int(sys.stdin.readline())
adj = [[] for _ in range(N + 1)]
T_prime_adj = [[] for _ in range(N // 2 + 1)]
T_prime_edge_map = {} # Maps T' edge {u_pair, v_pair} to T edge {u_vtx, v_vtx}

for i in range(1, N):
    u, v = map(int, sys.stdin.readline().split())
    adj[u].append(v)
    adj[v].append(u)
    if i >= N // 2:
        # This is an additional edge connecting two pairs
        pair_u = (u + 1) // 2
        pair_v = (v + 1) // 2
        if pair_u != pair_v: # Only add edge in T' if it connects different pairs
            T_prime_adj[pair_u].append(pair_v)
            T_prime_adj[pair_v].append(pair_u)
            # Store the T edge corresponding to the T' edge
            # Normalize T' edge by sorting pair indices
            p1, p2 = min(pair_u, pair_v), max(pair_u, pair_v)
            # Ensure the T edge is stored consistently with the T' edge direction
            # If pair_u < pair_v, store (u, v). If pair_v < pair_u, store (v, u).
            if pair_u < pair_v:
                 T_prime_edge_map[(p1, p2)] = (u, v)
            else:
                 T_prime_edge_map[(p1, p2)] = (v, u)


# 1. Color vertices and find v_i^0, v_i^1
color = [-1] * (N + 1)
q = [(1, 0)]
color[1] = 0
visited = [False] * (N + 1)
visited[1] = True
head = 0
while head < len(q):
    u, c = q[head]
    head += 1
    for v in adj[u]:
        if not visited[v]:
            visited[v] = True
            color[v] = 1 - c
            q.append((v, 1 - c))

v_colored = {} # v_colored[i][0] is vertex in pair i with color 0, v_colored[i][1] color 1
for i in range(1, N // 2 + 1):
    u, v = 2 * i - 1, 2 * i
    v_colored[i] = [0, 0]
    if color[u] == 0:
        v_colored[i][0] = u
        v_colored[i][1] = v
    else:
        v_colored[i][0] = v
        v_colored[i][1] = u

# Compute depths in T from root 1
depth = [-1] * (N + 1)
depth[1] = 0
q_depth = [1]
visited_depth = [False] * (N + 1)
visited_depth[1] = True
head_depth = 0
while head_depth < len(q_depth):
    u = q_depth[head_depth]
    head_depth += 1
    for v in adj[u]:
        if not visited_depth[v]:
            visited_depth[v] = True
            depth[v] = depth[u] + 1
            q_depth.append(v)

# 2. Root T' at 1 and perform DFS to build tree structure
parent_T_prime = [-1] * (N // 2 + 1)
T_prime_children = [[] for _ in range(N // 2 + 1)]
T_prime_dfs_stack = [(1, 0)]
visited_T_prime = [False] * (N // 2 + 1)
visited_T_prime[1] = True

while T_prime_dfs_stack:
    u, p = T_prime_dfs_stack.pop()
    parent_T_prime[u] = p
    # Use a set to handle multi-edges in T leading to same T' edge
    for v in set(T_prime_adj[u]):
        if v != p:
            T_prime_children[u].append(v)
            T_prime_dfs_stack.append((v, u))


results = []

# 3. DFS on T' to pair vertices
def dfs_pair(u):
    """
    Performs DFS on T' from node u. Pairs vertices locally and returns the single
    vertex from pair u that must be paired at the parent level (if u is not root).
    """
    available_here = [] # Vertices available for pairing at node u's level

    # Collect the single unmatched vertex returned from each child's subtree
    # This vertex is the one in the child pair connected to u's pair
    for v in T_prime_children[u]:
        unpaired_from_child = dfs_pair(v) # This returns a list containing 1 vertex
        available_here.extend(unpaired_from_child)

    if u == 1: # Root node
        # At root, both vertices from pair 1 are available
        available_here.append(v_colored[u][0])
        available_here.append(v_colored[u][1])
    else:
        # At non-root node, the vertex in pair u NOT connected to parent is available
        p = parent_T_prime[u]
        u_pair, p_pair = min(u, p), max(u, p)
        t_u_conn, t_p_conn = T_prime_edge_map[(u_pair, p_pair)]

        # Find which vertex in pair u is t_u_conn or t_p_conn
        vtx_in_u_conn_to_parent = t_u_conn if u_pair == u else t_p_conn

        other_vtx_in_pair = v_colored[u][0] if v_colored[u][0] != vtx_conn_to_parent else v_colored[u][1]
        available_here.append(other_vtx_in_pair)

    # available_here now contains vertices that need to be paired at this level
    # Size of available_here should be |children(u)| + 2 if u=1, and |children(u)| + 1 if u!=1
    # |children(u)| is the number of subtrees whose returned vertex is added
    # The number of vertices returned is |children(u)|.
    # Total size: |children(u)| + (1 if u != 1 else 2)

    # Sort by color (0 then 1), then depth descending to maximize distance
    available_here.sort(key=lambda v: (color[v], -depth[v]))

    # Pair up vertices: first half with second half
    # If u != 1, one vertex remains and is returned
    num_to_pair = len(available_here) if u == 1 else len(available_here) - 1
    k = num_to_pair // 2
    for i in range(k):
        results.append((available_here[i], available_here[i + k]))

    if u != 1:
         # The vertex connecting pair u to parent pair parent_T_prime[u] is the one left unmatched
         p = parent_T_prime[u]
         u_pair, p_pair = min(u, p), max(u, p)
         t_u_conn, t_p_conn = T_prime_edge_map[(u_pair, p_pair)]
         vtx_conn_to_parent = t_u_conn if u_pair == u else t_p_conn
         return [vtx_conn_to_parent] # Return the single vertex
    else:
        return [] # Root handles all pairings, nothing left


# Execute DFS from root 1
dfs_pair(1)

# Print results
for u, v in results:
    print(u, v)