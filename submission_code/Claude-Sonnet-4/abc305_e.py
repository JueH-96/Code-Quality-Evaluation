from collections import deque, defaultdict

# Read input
N, M, K = map(int, input().split())

# Build adjacency list
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

# Find all guarded vertices
guarded = set()

for guard_pos, stamina in guards:
    # BFS from guard position
    queue = deque([(guard_pos, 0)])  # (vertex, distance)
    visited = set([guard_pos])
    
    while queue:
        vertex, dist = queue.popleft()
        
        # This vertex is guarded by this guard
        guarded.add(vertex)
        
        # If we haven't reached stamina limit, explore neighbors
        if dist < stamina:
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

# Convert to sorted list
result = sorted(list(guarded))

# Output
print(len(result))
if result:
    print(*result)