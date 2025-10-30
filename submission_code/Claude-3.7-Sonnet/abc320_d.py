from collections import deque

def solve():
    N, M = map(int, input().split())
    
    # Create adjacency list for the directed graph
    graph = [[] for _ in range(N + 1)]
    
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        
        # From A's perspective, B is (X, Y) away
        graph[A].append((B, X, Y))
        
        # From B's perspective, A is (-X, -Y) away
        graph[B].append((A, -X, -Y))
    
    # Initialize coordinates
    coordinates = [None] * (N + 1)
    coordinates[1] = (0, 0)  # Person 1 is at the origin
    
    # Keep track of visited nodes
    visited = [False] * (N + 1)
    visited[1] = True
    
    # Start BFS from person 1
    queue = deque([1])
    
    while queue:
        person = queue.popleft()
        
        for neighbor, x_offset, y_offset in graph[person]:
            neighbor_x = coordinates[person][0] + x_offset
            neighbor_y = coordinates[person][1] + y_offset
            
            if not visited[neighbor]:
                coordinates[neighbor] = (neighbor_x, neighbor_y)
                visited[neighbor] = True
                queue.append(neighbor)
            elif coordinates[neighbor] != (neighbor_x, neighbor_y):
                # This shouldn't happen since information is consistent
                pass
    
    # Print results
    for i in range(1, N + 1):
        if coordinates[i] is not None:
            print(f"{coordinates[i][0]} {coordinates[i][1]}")
        else:
            print("undecidable")

solve()