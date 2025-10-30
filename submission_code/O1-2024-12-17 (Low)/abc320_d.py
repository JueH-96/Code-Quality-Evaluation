def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    
    edges = [[] for _ in range(N+1)]
    idx = 2
    for _ in range(M):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        X = int(input_data[idx+2]); Y = int(input_data[idx+3])
        idx += 4
        # Store the constraint: B = A + (X,Y) 
        edges[A].append((B, X, Y))
        # Also store the reverse: A = B - (X,Y)
        edges[B].append((A, -X, -Y))
    
    # We'll find coordinates relative to person 1, who is at (0,0).
    # Perform BFS/DFS from node 1 to find all reachable nodes and set coordinates.
    
    # Coordinates array (x, y) for each person; None if not determined yet.
    coords = [None] * (N+1)
    coords[1] = (0, 0)
    
    visited = [False] * (N+1)
    visited[1] = True
    
    queue = deque([1])
    
    while queue:
        cur = queue.popleft()
        cur_x, cur_y = coords[cur]
        for (nxt, dx, dy) in edges[cur]:
            if not visited[nxt]:
                visited[nxt] = True
                coords[nxt] = (cur_x + dx, cur_y + dy)
                queue.append(nxt)
            else:
                # If visited, input is guaranteed consistent, so no need to verify mismatch.
                pass
    
    # Output results
    # If a person's coordinate was never set (not in same component as 1), print "undecidable"
    # Otherwise, print the coordinate.
    for i in range(1, N+1):
        if coords[i] is None:
            print("undecidable")
        else:
            print(coords[i][0], coords[i][1])

# Do not forget to call main()
if __name__ == "__main__":
    main()