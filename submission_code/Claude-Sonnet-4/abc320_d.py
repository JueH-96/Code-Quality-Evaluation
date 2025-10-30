from collections import deque, defaultdict

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list for the graph
    graph = defaultdict(list)
    
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        # From A's perspective, B is at (X, Y)
        graph[A].append((B, X, Y))
        # From B's perspective, A is at (-X, -Y)
        graph[B].append((A, -X, -Y))
    
    # Initialize coordinates array
    coords = [None] * (N + 1)
    coords[1] = (0, 0)  # Person 1 is at origin
    
    # BFS to determine coordinates
    queue = deque([1])
    visited = set([1])
    
    while queue:
        current = queue.popleft()
        curr_x, curr_y = coords[current]
        
        for neighbor, dx, dy in graph[current]:
            new_x, new_y = curr_x + dx, curr_y + dy
            
            if neighbor not in visited:
                coords[neighbor] = (new_x, new_y)
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Output results
    for i in range(1, N + 1):
        if coords[i] is None:
            print("undecidable")
        else:
            print(coords[i][0], coords[i][1])

solve()