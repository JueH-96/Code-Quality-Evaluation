def solve():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and Q
    N, Q = map(int, data[0].split())
    
    # Initialize positions of the dragon parts
    positions = [(i, 0) for i in range(1, N + 1)]
    
    results = []
    
    for i in range(1, Q + 1):
        query = data[i].split()
        if query[0] == '1':
            direction = query[1]
            # Move the head
            head_x, head_y = positions[0]
            if direction == 'R':
                head_x += 1
            elif direction == 'L':
                head_x -= 1
            elif direction == 'U':
                head_y += 1
            elif direction == 'D':
                head_y -= 1
            
            # Update the head position
            positions[0] = (head_x, head_y)
            
            # Move the other parts
            for j in range(1, N):
                positions[j] = positions[j - 1]
        
        elif query[0] == '2':
            p = int(query[1]) - 1  # Convert to 0-based index
            results.append(f"{positions[p][0]} {positions[p][1]}")
    
    # Print all results for the second type of queries
    print("
".join(results))