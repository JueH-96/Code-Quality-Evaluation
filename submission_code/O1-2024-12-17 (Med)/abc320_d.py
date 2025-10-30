def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    
    # adjacency[u] will hold tuples of (v, dx, dy)
    # meaning pos[v] = pos[u] + (dx, dy)
    adjacency = [[] for _ in range(N+1)]
    idx = 2
    for _ in range(M):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        X = int(input_data[idx+2]); Y = int(input_data[idx+3])
        idx += 4
        # Store bidirectional info:
        # A -> B: (X, Y)
        # B -> A: (-X, -Y)
        adjacency[A].append((B, X, Y))
        adjacency[B].append((A, -X, -Y))
    
    # We'll use BFS starting from person 1
    visited = [False]*(N+1)
    pos = [None]*(N+1)
    
    # Person 1 is fixed at the origin
    pos[1] = (0, 0)
    visited[1] = True
    
    queue = deque([1])
    while queue:
        u = queue.popleft()
        ux, uy = pos[u]
        for (v, dx, dy) in adjacency[u]:
            # Proposed position for v
            vx = ux + dx
            vy = uy + dy
            if not visited[v]:
                visited[v] = True
                pos[v] = (vx, vy)
                queue.append(v)
            else:
                # The problem statement guarantees consistency,
                # so there's no need to check for conflicts here.
                pass
    
    # Output results
    # If a person is not visited (not in the same connected component as 1),
    # their position is undecidable
    for i in range(1, N+1):
        if visited[i]:
            print(pos[i][0], pos[i][1])
        else:
            print("undecidable")

# Do not forget to call main()
main()