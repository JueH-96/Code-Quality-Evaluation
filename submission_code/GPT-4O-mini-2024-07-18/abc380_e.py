def solve():
    import sys
    from collections import defaultdict, deque
    
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and Q
    N, Q = map(int, data[0].split())
    
    # Initialize the colors of the cells
    colors = list(range(1, N + 1))
    
    # To keep track of the count of each color
    color_count = defaultdict(int)
    for color in colors:
        color_count[color] += 1
    
    results = []
    
    for i in range(1, Q + 1):
        query = list(map(int, data[i].split()))
        
        if query[0] == 1:  # Repaint operation
            x, c = query[1] - 1, query[2]  # Convert to 0-based index
            original_color = colors[x]
            
            if original_color != c:
                # BFS or DFS to find all reachable cells with the same original color
                queue = deque([x])
                visited = set()
                visited.add(x)
                
                while queue:
                    current = queue.popleft()
                    colors[current] = c  # Repaint the cell
                    color_count[c] += 1  # Increment the count of new color
                    color_count[original_color] -= 1  # Decrement the count of old color
                    
                    # Check adjacent cells
                    if current > 0 and colors[current - 1] == original_color and (current - 1) not in visited:
                        visited.add(current - 1)
                        queue.append(current - 1)
                    if current < N - 1 and colors[current + 1] == original_color and (current + 1) not in visited:
                        visited.add(current + 1)
                        queue.append(current + 1)
        
        elif query[0] == 2:  # Count operation
            c = query[1]
            results.append(color_count[c])
    
    # Print all results for the count queries
    sys.stdout.write('
'.join(map(str, results)) + '
')