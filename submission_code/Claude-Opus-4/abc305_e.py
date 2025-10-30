from collections import deque

# Read input
N, M, K = map(int, input().split())

# Build adjacency list for the graph
graph = [[] for _ in range(N + 1)]
for _ in range(M):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)

# Read guards information
guards = []
for _ in range(K):
    p, h = map(int, input().split())
    guards.append((p, h))

# Find all guarded vertices
guarded = set()

# For each guard, do BFS to find all vertices within distance h
for guard_pos, stamina in guards:
    # BFS from guard position
    visited = [False] * (N + 1)
    queue = deque([(guard_pos, 0)])  # (vertex, distance)
    visited[guard_pos] = True
    
    while queue:
        vertex, dist = queue.popleft()
        
        # If this vertex is within stamina distance, it's guarded
        if dist <= stamina:
            guarded.add(vertex)
        
        # If we can still go further
        if dist < stamina:
            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))

# Convert to sorted list
guarded_list = sorted(list(guarded))

# Output
print(len(guarded_list))
if guarded_list:
    print(*guarded_list)