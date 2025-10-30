from collections import defaultdict, deque

def min_operations(N, M, S, T, edges):
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # BFS to find the shortest path
    queue = deque([(S, T, 0)])  # (pos_A, pos_B, distance)
    visited = {(S, T)}
    
    while queue:
        pos_A, pos_B, distance = queue.popleft()
        
        # Check if we've reached the goal state
        if pos_A == T and pos_B == S:
            return distance
        
        # Try moving piece A
        for next_A in graph[pos_A]:
            if next_A != pos_B:  # Can't have both pieces on the same vertex
                new_state = (next_A, pos_B)
                if new_state not in visited:
                    queue.append((next_A, pos_B, distance + 1))
                    visited.add(new_state)
        
        # Try moving piece B
        for next_B in graph[pos_B]:
            if next_B != pos_A:  # Can't have both pieces on the same vertex
                new_state = (pos_A, next_B)
                if new_state not in visited:
                    queue.append((pos_A, next_B, distance + 1))
                    visited.add(new_state)
    
    return -1  # Goal state not reachable

# Read input
N, M, S, T = map(int, input().split())
edges = []
for _ in range(M):
    u, v = map(int, input().split())
    edges.append((u, v))

# Output the result
print(min_operations(N, M, S, T, edges))