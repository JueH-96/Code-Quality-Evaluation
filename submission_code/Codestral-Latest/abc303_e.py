import sys
from collections import defaultdict, deque

def find_stars(N, edges):
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Find the leaves (vertices with degree 1)
    leaves = [v for v in graph if len(graph[v]) == 1]

    # BFS to find the levels of the stars
    levels = []
    while leaves:
        leaf = leaves.pop()
        level = 1
        queue = deque([(leaf, 1)])
        visited = set()
        while queue:
            node, dist = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            level = max(level, dist)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))
        levels.append(level)
        # Remove the star from the graph
        for node in visited:
            for neighbor in graph[node]:
                graph[neighbor].remove(node)
            del graph[node]
        # Update the leaves
        leaves = [v for v in graph if len(graph[v]) == 1]

    # Sort the levels in ascending order
    levels.sort()
    return levels

# Read input
N = int(sys.stdin.readline().strip())
edges = []
for _ in range(N - 1):
    u, v = map(int, sys.stdin.readline().strip().split())
    edges.append((u, v))

# Find and print the levels of the stars
levels = find_stars(N, edges)
print(" ".join(map(str, levels)))