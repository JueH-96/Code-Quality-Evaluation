from collections import defaultdict, deque

def solve():
    N, M = map(int, input().split())
    
    # Build graph
    graph = defaultdict(list)
    constraints = {}
    
    for _ in range(M):
        x, y, z = map(int, input().split())
        x -= 1  # Convert to 0-indexed
        y -= 1
        graph[x].append((y, z))
        graph[y].append((x, z))
        constraints[(min(x, y), max(x, y))] = z
    
    A = [-1] * N  # -1 means unassigned
    
    # Process each connected component
    for start in range(N):
        if A[start] != -1:
            continue
            
        # BFS to assign values in this component
        queue = deque([start])
        A[start] = 0  # Set root to 0 for minimum sum
        
        while queue:
            u = queue.popleft()
            
            for v, z in graph[u]:
                expected_v = A[u] ^ z
                
                if A[v] == -1:
                    A[v] = expected_v
                    queue.append(v)
                elif A[v] != expected_v:
                    # Contradiction found
                    print(-1)
                    return
    
    # Verify all constraints are satisfied
    for _ in range(M):
        pass  # We already read all constraints above
    
    # Re-read constraints to verify (since we consumed input)
    # Actually, let's verify using our stored constraints
    for (u, v), z in constraints.items():
        if A[u] ^ A[v] != z:
            print(-1)
            return
    
    print(*A)

# Read input and solve
N, M = map(int, input().split())

# Build graph
graph = defaultdict(list)
edges = []

for _ in range(M):
    x, y, z = map(int, input().split())
    x -= 1  # Convert to 0-indexed
    y -= 1
    graph[x].append((y, z))
    if x != y:  # Avoid duplicate edges for self-loops
        graph[y].append((x, z))
    edges.append((x, y, z))

A = [-1] * N  # -1 means unassigned

# Process each connected component
for start in range(N):
    if A[start] != -1:
        continue
        
    # BFS to assign values in this component
    queue = deque([start])
    A[start] = 0  # Set root to 0 for minimum sum
    
    valid = True
    while queue and valid:
        u = queue.popleft()
        
        for v, z in graph[u]:
            expected_v = A[u] ^ z
            
            if A[v] == -1:
                A[v] = expected_v
                queue.append(v)
            elif A[v] != expected_v:
                # Contradiction found
                valid = False
                break
    
    if not valid:
        print(-1)
        exit()

# Verify all original constraints
for x, y, z in edges:
    if A[x] ^ A[y] != z:
        print(-1)
        exit()

print(*A)