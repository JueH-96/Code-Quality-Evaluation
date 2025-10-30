import sys
from collections import deque

# Read all input data
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Initialize adjacency lists for original and reverse graphs
adj_orig = [[] for _ in range(N + 1)]
adj_rev = [[] for _ in range(N + 1)]

# Read dependencies and build graphs
for i in range(1, N + 1):
    C = int(data[index])
    index += 1
    for _ in range(C):
        p = int(data[index])
        index += 1
        # Original graph: edge from prerequisite p to book i
        adj_orig[p].append(i)
        # Reverse graph: edge from book i to prerequisite p
        adj_rev[i].append(p)

# BFS to find all nodes reachable from 1 in the reverse graph
vis = [False] * (N + 1)
queue_bfs = deque()
queue_bfs.append(1)
vis[1] = True

while queue_bfs:
    node = queue_bfs.popleft()
    for nei in adj_rev[node]:
        if not vis[nei]:
            vis[nei] = True
            queue_bfs.append(nei)

# Kahn's algorithm for topological sort on the subgraph
indegree = [0] * (N + 1)

# Compute indegree for nodes in the subgraph (where vis[node] is True)
for u in range(1, N + 1):
    if vis[u]:
        for v in adj_orig[u]:
            if vis[v]:
                indegree[v] += 1

# Initialize queue with nodes having indegree 0 in the subgraph
queue_topo = deque()
for node in range(1, N + 1):
    if vis[node] and indegree[node] == 0:
        queue_topo.append(node)

# Perform topological sort
order = []
while queue_topo:
    curr = queue_topo.popleft()
    order.append(curr)
    for nei in adj_orig[curr]:
        if vis[nei]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                queue_topo.append(nei)

# The order list contains all nodes in the subgraph in topological order
# The last node is book 1, so exclude it and output the rest
result = order[:-1]
print(' '.join(map(str, result)))