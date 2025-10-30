def solve():
    N, M = map(int, input().split())
    
    # Store relative positions
    edges = []
    for _ in range(M):
        a, b, x, y = map(int, input().split())
        edges.append((a-1, b-1, x, y))
    
    # Try to determine coordinates for each person
    coords = [(0,0)] + [(None,None)]*(N-1) # Person 1 is at origin
    
    # Keep trying to determine coordinates until no more progress
    changed = True
    while changed:
        changed = False
        for a, b, x, y in edges:
            # If we know position of a but not b
            if coords[a] != (None,None) and coords[b] == (None,None):
                ax,ay = coords[a]
                coords[b] = (ax+x, ay+y)
                changed = True
            # If we know position of b but not a    
            elif coords[b] != (None,None) and coords[a] == (None,None):
                bx,by = coords[b]
                coords[a] = (bx-x, by-y)
                changed = True
    
    # Verify uniqueness by checking all constraints
    valid = [True] * N
    for a, b, x, y in edges:
        if coords[a] != (None,None) and coords[b] != (None,None):
            ax,ay = coords[a]
            bx,by = coords[b]
            if bx-ax != x or by-ay != y:
                valid[a] = valid[b] = False

    # Output results
    for i in range(N):
        if coords[i] == (None,None) or not valid[i]:
            print("undecidable")
        else:
            print(f"{coords[i][0]} {coords[i][1]}")

solve()