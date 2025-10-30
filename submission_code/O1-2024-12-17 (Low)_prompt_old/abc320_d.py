def solve():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])

    # Edge cases
    if N == 1:
        # Only person 1 exists, at the origin
        print("0 0")
        return

    # Build adjacency list
    # adj[u] will hold tuples (v, dx, dy) meaning:
    # from person u to person v, the offset is (dx, dy), i.e.  v = u + (dx, dy)
    adj = [[] for _ in range(N+1)]
    idx = 2
    for _ in range(M):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        X = int(input_data[idx+2]); Y = int(input_data[idx+3])
        idx += 4
        # Add edge A->B
        adj[A].append((B, X, Y))
        # Add edge B->A (reverse offset)
        adj[B].append((A, -X, -Y))

    # We'll keep track of whether we've visited a person (their position is fixed)
    visited = [False]*(N+1)
    # pos[i] will be a tuple (x, y) for the coordinate of person i
    pos = [None]*(N+1)

    # Person 1 is at (0,0)
    visited[1] = True
    pos[1] = (0, 0)

    # BFS/DFS from person 1
    queue = deque([1])
    while queue:
        u = queue.popleft()
        ux, uy = pos[u]
        for (v, dx, dy) in adj[u]:
            if not visited[v]:
                visited[v] = True
                # v = u + (dx, dy)
                pos[v] = (ux + dx, uy + dy)
                queue.append(v)
            else:
                # The problem guarantees consistency, so no need to check
                pass

    # Output results
    for i in range(1, N+1):
        if visited[i]:
            print(pos[i][0], pos[i][1])
        else:
            print("undecidable")

# Call solve() after defining it
solve()