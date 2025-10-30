def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Convert to 0-indexed
    edges = [a[i] - 1 for i in range(n)]
    
    total_pairs = 0
    
    for start in range(n):
        visited = set()
        current = start
        path = []
        
        # Follow the path until we find a cycle
        while current not in visited:
            visited.add(current)
            path.append(current)
            current = edges[current]
        
        # Find where the cycle starts
        cycle_start_idx = path.index(current)
        
        # Everything from cycle_start_idx onwards is in the cycle
        cycle_vertices = set(path[cycle_start_idx:])
        
        # Vertices reachable from start vertex:
        # - All vertices on the path (including start itself)
        # - All vertices in the cycle
        reachable = set(path) | cycle_vertices
        
        total_pairs += len(reachable)
    
    print(total_pairs)

solve()