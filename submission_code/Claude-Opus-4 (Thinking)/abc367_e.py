n, k = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

if k == 0:
    print(' '.join(map(str, a)))
else:
    result = [0] * n
    
    for i in range(n):
        # Find where position i gets its value from after k operations
        visited = {}
        path = []
        current = i
        
        # Build path until we find a cycle
        while current not in visited:
            visited[current] = len(path)
            path.append(current)
            current = x[current] - 1  # Convert to 0-indexed
        
        # Detect cycle
        cycle_start = visited[current]
        cycle_length = len(path) - cycle_start
        
        # Determine source position after k steps
        if k < len(path):
            # Still in the initial path
            source = path[k]
        else:
            # In the cycle
            k_in_cycle = (k - cycle_start) % cycle_length
            source = path[cycle_start + k_in_cycle]
        
        result[i] = a[source]
    
    print(' '.join(map(str, result)))