import heapq
import math
from collections import deque
import sys
sys.setrecursionlimit(1000005)  # Increase recursion limit for DFS
INF = math.inf

# Define Dijkstra function
def dijkstra(adj, start, num_nodes):
    dist = [INF] * (num_nodes + 1)
    dist[start] = 0
    pq = [(0, start)]  # distance, node
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:  # outdated entry
            continue
        for v, w in adj[u]:
            if dist[v] > d + w:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist

# Define find_bridges function
def find_bridges(g_adj, num_nodes):
    disc = [-1] * (num_nodes + 1)
    low = [-1] * (num_nodes + 1)
    parent_node = [-1] * (num_nodes + 1)
    bridges = []
    timer_counter = [0]  # Use list to allow modification
    def dfs(u):
        disc[u] = timer_counter[0]
        low[u] = timer_counter[0]
        timer_counter[0] += 1
        for v, edge_idx in g_adj[u]:
            if disc[v] == -1:  # not visited
                parent_node[v] = u
                dfs(v)
                if low[v] > disc[u]:
                    bridges.append(edge_idx)
                low[u] = min(low[u], low[v])
            elif v != parent_node[u]:  # back edge
                low[u] = min(low[u], disc[v])
    for i in range(1, num_nodes + 1):
        if disc[i] == -1:
            dfs(i)
    return bridges

# Define find_components function
def find_components(non_bridge_adj, num_nodes):
    visited = [False] * (num_nodes + 1)
    comp_list = [-1] * (num_nodes + 1)
    cid = 0
    for i in range(1, num_nodes + 1):
        if not visited[i]:
            queue = deque([i])
            visited[i] = True
            comp_list[i] = cid
            while queue:
                node = queue.popleft()
                for nei in non_bridge_adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        comp_list[nei] = cid
                        queue.append(nei)
            cid += 1
    return comp_list, cid

# Define find_critical_edges function
def find_critical_edges(bridge_tree_adj, start_comp, end_comp, num_comp):
    visited = [False] * num_comp
    parent_comp = [-1] * num_comp
    parent_edge_idx = [-1] * num_comp
    queue = deque()
    queue.append(start_comp)
    visited[start_comp] = True
    parent_comp[start_comp] = -1
    while queue:
        current = queue.popleft()
        for nei_comp, edge_idx in bridge_tree_adj[current]:
            if not visited[nei_comp]:
                visited[nei_comp] = True
                parent_comp[nei_comp] = current
                parent_edge_idx[nei_comp] = edge_idx
                queue.append(nei_comp)
    # Now, if end_comp is visited, reconstruct path
    if not visited[end_comp]:
        return set()  # should not happen
    critical_edges_set = set()
    current = end_comp
    while current != start_comp:
        edge_idx = parent_edge_idx[current]
        critical_edges_set.add(edge_idx)
        current = parent_comp[current]
    return critical_edges_set

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
M = int(data[index + 1])
index += 2
A_list = [0] * M
B_list = [0] * M
C_list = [0] * M
for i in range(M):
    A = int(data[index])
    B = int(data[index + 1])
    C = int(data[index + 2])
    index += 3
    A_list[i] = A
    B_list[i] = B
    C_list[i] = C

# Build adjacency list for Dijkstra
adj_dijkstra = [[] for _ in range(N + 1)]
for i in range(M):
    adj_dijkstra[A_list[i]].append((B_list[i], C_list[i]))
    adj_dijkstra[B_list[i]].append((A_list[i], C_list[i]))

# Run Dijkstra
dist_from_1 = dijkstra(adj_dijkstra, 1, N)
dist_to_N = dijkstra(adj_dijkstra, N, N)  # from N
dist_all = dist_from_1[N]

# Find G_sp edges
is_gsp = [False] * M
g_sp_adj = [[] for _ in range(N + 1)]  # (nei, edge_idx)
for i in range(M):
    A = A_list[i]
    B = B_list[i]
    C = C_list[i]
    if dist_from_1[A] + C + dist_to_N[B] == dist_all or dist_from_1[B] + C + dist_to_N[A] == dist_all:
        is_gsp[i] = True
        g_sp_adj[A].append((B, i))
        g_sp_adj[B].append((A, i))

# Find bridges in G_sp
bridges = find_bridges(g_sp_adj, N)
bridge_set = set(bridges)

# Build non_bridge_adj
non_bridge_adj = [[] for _ in range(N + 1)]
for u in range(1, N + 1):
    for v, edge_idx in g_sp_adj[u]:
        if u < v and edge_idx not in bridge_set:  # add only when u < v
            non_bridge_adj[u].append(v)
            non_bridge_adj[v].append(u)

# Find components after removing bridges
comp_list, num_comp = find_components(non_bridge_adj, N)

# comp_s and comp_t
comp_s = comp_list[1]
comp_t = comp_list[N]

# Build bridge tree adjacency list
bridge_tree_adj = [[] for _ in range(num_comp)]  # list of (nei_comp, edge_idx)
for edge_idx in bridges:  # for each bridge
    u = A_list[edge_idx]
    v = B_list[edge_idx]
    comp_u = comp_list[u]
    comp_v = comp_list[v]
    bridge_tree_adj[comp_u].append((comp_v, edge_idx))
    bridge_tree_adj[comp_v].append((comp_u, edge_idx))

# Find critical edges
critical_edges_set = find_critical_edges(bridge_tree_adj, comp_s, comp_t, num_comp)

# Output
for i in range(M):  # i from 0 to M-1
    if is_gsp[i] and i in critical_edges_set:
        print("Yes")
    else:
        print("No")