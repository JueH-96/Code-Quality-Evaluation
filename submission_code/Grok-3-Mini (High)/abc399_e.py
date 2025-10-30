import sys
from collections import defaultdict

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
S = data[index]
index += 1
T = data[index]

# Check consistency of T for each S character
d_target = {}
for i in range(N):
    s_c = S[i]
    t_c = T[i]
    if s_c not in d_target:
        d_target[s_c] = t_c
    elif d_target[s_c] != t_c:
        print(-1)
        sys.exit()

# Find V: characters with s_c != target
V = [c for c in d_target if d_target[c] != c]

if not V:
    print(0)
    sys.exit()

# Build undirected adjacency for weak components
adj_undirected = defaultdict(set)
for c in V:
    target = d_target[c]
    if target in V:
        adj_undirected[c].add(target)
        adj_undirected[target].add(c)

# Find connected components using DFS
visited = set()
components = []

def dfs(node, adj, comp):
    stack = [node]
    visited.add(node)
    comp.append(node)
    while stack:
        current = stack[-1]
        found_new = False
        for neighbor in list(adj[current]):  # Use list to avoid runtime error
            if neighbor not in visited:
                visited.add(neighbor)
                comp.append(neighbor)
                stack.append(neighbor)
                found_new = True
                break  # Add one neighbor at a time for DFS
        if not found_new:
            stack.pop()

for node in V:
    if node not in visited:
        component = []
        dfs(node, adj_undirected, component)
        components.append(component)

# Calculate min operations
total_min_ops = 0
for comp in components:
    if not comp:
        continue
    # Check for cycle starting from first node in component
    start = comp[0]
    has_cycle = False
    path_set = set()
    current = start
    while True:
        if current in path_set:
            has_cycle = True
            break
        path_set.add(current)
        next_node = d_target[current]
        if next_node not in V:
            break  # No outedge to V
        current = next_node
    size_comp = len(comp)
    if has_cycle:
        min_ops_comp = size_comp + 1
    else:
        min_ops_comp = size_comp
    total_min_ops += min_ops_comp

print(total_min_ops)