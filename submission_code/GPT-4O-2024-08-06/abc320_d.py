# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    from collections import deque, defaultdict
    
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    edges = defaultdict(list)
    
    index = 2
    for _ in range(M):
        A = int(data[index]) - 1
        B = int(data[index + 1]) - 1
        X = int(data[index + 2])
        Y = int(data[index + 3])
        edges[A].append((B, X, Y))
        index += 4
    
    # Initialize coordinates
    coordinates = [None] * N
    coordinates[0] = (0, 0)  # Person 1 is at the origin
    
    # BFS to determine coordinates
    queue = deque([0])
    
    while queue:
        current = queue.popleft()
        current_x, current_y = coordinates[current]
        
        for neighbor, dx, dy in edges[current]:
            if coordinates[neighbor] is None:
                # Calculate the coordinates of the neighbor
                coordinates[neighbor] = (current_x + dx, current_y + dy)
                queue.append(neighbor)
    
    # Output the results
    result = []
    for i in range(N):
        if coordinates[i] is None:
            result.append("undecidable")
        else:
            result.append(f"{coordinates[i][0]} {coordinates[i][1]}")
    
    print("
".join(result))

main()