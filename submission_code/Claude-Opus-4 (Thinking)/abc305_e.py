from collections import deque, defaultdict

# Read input
N, M, K = map(int, input().split())

# Build the graph
graph = defaultdict(list)
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# Read guard positions and staminas
guards = []
for _ in range(K):
    p, h = map(int, input().split())
    guards.append((p, h))

# Find all guarded vertices
guarded = set()

for p, h in guards:
    # BFS from vertex p
    visited = {p}
    queue = deque([(p, 0)])  # (vertex, distance)
    
    while queue:
        v, dist = queue.popleft()
        
        # If distance is within stamina, this vertex is guarded
        if dist <= h:
            guarded.add(v)
        
        # If we've reached maximum distance, don't explore further
        if dist < h:
            for neighbor in graph[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

# Sort and output guarded vertices
guarded_list = sorted(list(guarded))
print(len(guarded_list))
print(' '.join(map(str, guarded_list)))