from collections import deque, defaultdict

def solve():
    # Read input
    N, M, S, T = map(int, input().split())
    
    # Build adjacency list
    graph = defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    # BFS to find minimum operations
    # State: (pos_A, pos_B, operations)
    queue = deque([(S, T, 0)])
    visited = set()
    visited.add((S, T))
    
    while queue:
        pos_A, pos_B, ops = queue.popleft()
        
        # Check if we reached the goal
        if pos_A == T and pos_B == S:
            return ops
        
        # Try moving piece A
        for next_A in graph[pos_A]:
            if next_A != pos_B and (next_A, pos_B) not in visited:
                visited.add((next_A, pos_B))
                queue.append((next_A, pos_B, ops + 1))
        
        # Try moving piece B
        for next_B in graph[pos_B]:
            if next_B != pos_A and (pos_A, next_B) not in visited:
                visited.add((pos_A, next_B))
                queue.append((pos_A, next_B, ops + 1))
    
    return -1

print(solve())