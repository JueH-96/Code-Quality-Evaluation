from collections import defaultdict, deque

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list with relative positions
    graph = defaultdict(list)
    
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        graph[A].append((B, X, Y))
    
    # Initialize positions - None means undecidable
    positions = [None] * (N + 1)
    positions[1] = (0, 0)  # Person 1 is at origin
    
    # BFS to find all reachable people from person 1
    queue = deque([1])
    
    while queue:
        current = queue.popleft()
        curr_x, curr_y = positions[current]
        
        for neighbor, dx, dy in graph[current]:
            if positions[neighbor] is None:
                positions[neighbor] = (curr_x + dx, curr_y + dy)
                queue.append(neighbor)
    
    # Output results
    for i in range(1, N + 1):
        if positions[i] is None:
            print("undecidable")
        else:
            print(positions[i][0], positions[i][1])

solve()