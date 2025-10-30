def find_cycle():
    N = int(input())
    A = [int(x) for x in input().split()]
    A = [x-1 for x in A]  # Convert to 0-based indexing
    
    # For each vertex, find a cycle starting from it
    for start in range(N):
        visited = set()
        curr = start
        path = []
        
        # Follow edges until we find a cycle
        while curr not in visited:
            visited.add(curr)
            path.append(curr)
            curr = A[curr]
            
            # If we hit a vertex in our current path
            if curr in visited:
                # Find where the cycle starts
                cycle_start = path.index(curr)
                cycle = path[cycle_start:]
                
                # Convert back to 1-based indexing and print
                print(len(cycle))
                print(' '.join(str(x+1) for x in cycle))
                return
                
# Read input and solve
find_cycle()