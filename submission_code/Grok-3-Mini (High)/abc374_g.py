import sys
sys.setrecursionlimit(10000)
from collections import defaultdict

def dfs_finish(graph, node, visited, order):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_finish(graph, neighbor, visited, order)
    order.append(node)

def dfs_component(graph, start, visited):
    component = []
    stack = [start]
    visited.add(start)
    while stack:
        current = stack.pop()
        component.append(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return component

def find_connected_components(graph_undir_dict, all_nodes):
    visited = set()
    components = []
    for node in all_nodes:
        if node not in visited:
            comp = dfs_component(graph_undir_dict, node, visited)
            components.append(comp)
    return components

data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
edges = []
for i in range(N):
    e = data[index]
    index += 1
    edges.append(e)

# Collect all nodes
all_nodes_set = set()
for e in edges:
    all_nodes_set.add(e[0])
    all_nodes_set.add(e[1])
all_nodes = list(all_nodes_set)

# Create adjacency list and transpose
adj = defaultdict(list)
adj_t = defaultdict(list)
for e in edges:
    u, v = e[0], e[1]
    adj[u].append(v)
    adj_t[v].append(u)

# First DFS for finish times
visited = set()
order = []
for node in all_nodes:
    if node not in visited:
        dfs_finish(adj, node, visited, order)

# Reverse for decreasing finish time
finished_order = list(reversed(order))

# Reset visited
visited = set()

# Second DFS on transpose for SCC
scc_list = []
for node in finished_order:
    if node not in visited:
        component = dfs_component(adj_t, node, visited)
        scc_list.append(component)

M = len(scc_list)

# Map node to SCC index
scc_index = {}
for i, comp in enumerate(scc_list):
    for n in comp:
        scc_index[n] = i

# Check for internal edges in each SCC
has_internal_edges = [False] * M
for e in edges:
    u, v = e[0], e[1]
    scc_u = scc_index[u]
    scc_v = scc_index[v]
    if scc_u == scc_v:
        has_internal_edges[scc_u] = True

# Create condensation adjacency list
cond_adj = defaultdict(set)
for e in edges:
    u_str, v_str = e[0], e[1]
    scc_u = scc_index[u_str]
    scc_v = scc_index[v_str]
    if scc_u != scc_v:
        cond_adj[scc_u].add(scc_v)

# Create undirected edge set for condensation weakly connected components
undirected_edge_set = set()
for u in range(M):
    for v in cond_adj[u]:
        pair = tuple(sorted([u, v]))
        undirected_edge_set.add(pair)

# Create undirected graph for condensation
graph_undir_dict = defaultdict(set)
for edge in undirected_edge_set:
    i, j = edge
    graph_undir_dict[i].add(j)
    graph_undir_dict[j].add(i)

# Find weakly connected components of condensation graph
comp_list = find_connected_components(graph_undir_dict, list(range(M)))

# Compute answer
answer = 0
for comp in comp_list:  # comp is list of SCC indices
    comp_set = set(comp)
    # Check if there are arcs within comp
    has_arcs = False
    for u in comp_set:
        for v in cond_adj[u]:
            if v in comp_set:
                has_arcs = True
                break
        if has_arcs:
            break
    if has_arcs:
        # Compute max outdegree within comp
        max_out = 0
        for u in comp_set:
            out_deg = sum(1 for v in cond_adj[u] if v in comp_set)
            if out_deg > max_out:
                max_out = out_deg
        answer += max_out
    else:
        # No arcs, comp has one SCC index
        scc_id = comp[0]  # Safe, size should be 1
        if has_internal_edges[scc_id]:
            answer += 1

print(answer)