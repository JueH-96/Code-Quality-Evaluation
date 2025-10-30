import sys
from collections import deque

# Read all input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1

# Build adjacency list
adj = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    u = int(data[index])
    v = int(data[index + 1])
    index += 2
    adj[u].append(v)
    adj[v].append(u)

# Find vertices with degree exactly 3
deg3_vertices = [i for i in range(1, N + 1) if len(adj[i]) == 3]

# Visited array for BFS
vis = [False] * (N + 1)

# Answer variable
answer = 0

# Iterate through all degree 3 vertices to find components
for start in deg3_vertices:
    if not vis[start]:
        # BFS to find all vertices in this component
        component = []
        queue = deque()
        queue.append(start)
        vis[start] = True
        component.append(start)
        
        while queue:
            current = queue.popleft()
            for nei in adj[current]:
                if len(adj[nei]) == 3 and not vis[nei]:
                    vis[nei] = True
                    queue.append(nei)
                    component.append(nei)
        
        # Find all degree 2 vertices attached to this component
        attached_set = set()
        for vert in component:
            for nei in adj[vert]:
                if len(adj[nei]) == 2:
                    attached_set.add(nei)
        
        # Number of attached degree 2 vertices
        k = len(attached_set)
        
        # Add binomial coefficient C(k, 2) to answer
        answer += k * (k - 1) // 2

# Output the answer
print(answer)