from collections import defaultdict, deque

n = int(input())
edges = []
for _ in range(n - 1):
    a, b = map(int, input().split())
    edges.append((a, b))

# Build adjacency list
adj = defaultdict(list)
for a, b in edges:
    adj[a].append(b)
    adj[b].append(a)

# Track removed vertices
removed = set()
result = []

def find_leaves():
    leaves = []
    for v in range(1, n + 1):
        if v not in removed:
            degree = sum(1 for u in adj[v] if u not in removed)
            if degree == 1:
                leaves.append(v)
    return leaves

def bfs_distance(start, end):
    if start == end:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        node, dist = queue.popleft()
        for neighbor in adj[node]:
            if neighbor in removed:
                continue
            if neighbor == end:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1

for _ in range(n // 2):
    leaves = find_leaves()
    
    # Special case: only 2 vertices left
    remaining = [v for v in range(1, n + 1) if v not in removed]
    if len(remaining) == 2:
        result.append((remaining[0], remaining[1]))
        break
    
    # Find the pair with maximum distance
    max_dist = -1
    best_pair = None
    
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            dist = bfs_distance(leaves[i], leaves[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (leaves[i], leaves[j])
    
    if best_pair:
        result.append(best_pair)
        removed.add(best_pair[0])
        removed.add(best_pair[1])

for a, b in result:
    print(a, b)