import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    from collections import defaultdict, deque
    
    # Relations from A to B with offsets X, Y
    relations = defaultdict(list)
    
    for _ in range(M):
        A = int(data[index]) - 1
        index += 1
        B = int(data[index]) - 1
        index += 1
        X = int(data[index])
        index += 1
        Y = int(data[index])
        index += 1
        relations[A].append((B, X, Y))
        relations[B].append((A, -X, -Y))
    
    # To store the coordinates of each person
    coordinates = [None] * N
    # To check if the person's coordinates are undecidable
    undecidable = [False] * N
    
    # BFS to determine coordinates
    def bfs(start):
        queue = deque([start])
        coordinates[start] = (0, 0)  # Start person is at the origin
        visited = set([start])
        
        while queue:
            current = queue.popleft()
            current_x, current_y = coordinates[current]
            
            for neighbor, dx, dy in relations[current]:
                neighbor_x = current_x + dx
                neighbor_y = current_y + dy
                
                if neighbor in visited:
                    # If already visited, check for consistency
                    if coordinates[neighbor] != (neighbor_x, neighbor_y):
                        undecidable[neighbor] = True
                else:
                    # Set the coordinates and mark as visited
                    coordinates[neighbor] = (neighbor_x, neighbor_y)
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    # We need to check each component
    for i in range(N):
        if coordinates[i] is None:
            bfs(i)
    
    # Output the results
    for i in range(N):
        if undecidable[i]:
            print("undecidable")
        elif coordinates[i] is not None:
            print(f"{coordinates[i][0]} {coordinates[i][1]}")
        else:
            print("undecidable")