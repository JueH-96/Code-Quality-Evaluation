from collections import defaultdict, deque

# Read input
N, M, K = map(int, input().split())

# Construct the graph
graph = defaultdict(list)
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# Read guard information
guards = []
for _ in range(K):
    p, h = map(int, input().split())
    guards.append((p, h))

# Use a set to keep track of guarded vertices
guarded_vertices = set()

# For each guard, perform BFS to determine guarded vertices
for p, h in guards:
    queue = deque([(p, 0)])  # (vertex, distance)
    visited = set([p])
    
    while queue:
        vertex, distance = queue.popleft()
        
        if distance <= h:
            guarded_vertices.add(vertex)
        
        if distance < h:  # We can still move further
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

# Sort guarded vertices
guarded_vertices = sorted(list(guarded_vertices))

# Output
print(len(guarded_vertices))
print(' '.join(map(str, guarded_vertices)))