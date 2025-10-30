import sys
from collections import deque

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # adj[i] is a list of tuples (neighbor_idx, dx, dy) for person i (0-indexed)
    adj = [[] for _ in range(N)] 
    for _ in range(M):
        A, B, X, Y = map(int, sys.stdin.readline().split())
        # Convert 1-indexed A, B to 0-indexed A_idx, B_idx
        A_idx = A - 1
        B_idx = B - 1
        
        # Edge from A_idx to B_idx: P_B = P_A + (X,Y)
        adj[A_idx].append((B_idx, X, Y))
        # Edge from B_idx to A_idx: P_A = P_B + (-X,-Y)
        adj[B_idx].append((A_idx, -X, -Y))

    # positions[i] stores (x_coord, y_coord) for person i (0-indexed)
    # Initialize all to None, indicating coordinates are not yet determined
    positions = [None] * N
    
    # Person 1 (index 0) is at the origin (0,0)
    # N >= 1 is guaranteed by problem constraints, so person 0 (person 1 in 1-indexed) always exists.
    positions[0] = (0, 0)
    
    # Queue for BFS. Stores 0-indexed person IDs whose positions are known
    # and whose neighbors need to be processed.
    q = deque()
    q.append(0) # Start BFS from person 0
    
    while q:
        u_idx = q.popleft()
        ux, uy = positions[u_idx]
        
        for v_idx, dx, dy in adj[u_idx]:
            # If v_idx's position is not yet determined
            if positions[v_idx] is None:
                positions[v_idx] = (ux + dx, uy + dy)
                q.append(v_idx)
            # Else (positions[v_idx] is already known):
            #   The problem guarantees consistency. This means any path from person 0
            #   to v_idx will result in the same coordinates. So, if we've already
            #   found coordinates for v_idx, we don't need to do anything more with this path.

    # Prepare output strings
    output_lines = []
    for i in range(N): # For each person (0-indexed i corresponds to person i+1)
        if positions[i] is None:
            output_lines.append("undecidable")
        else:
            # Person i (0-indexed) has coordinates positions[i] = (x, y)
            # Output format requires "x y"
            output_lines.append(f"{positions[i][0]} {positions[i][1]}")
            
    # Print all results, each on a new line
    sys.stdout.write("
".join(output_lines) + "
")

if __name__ == '__main__':
    solve()