def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    
    # Adjacency list: for each person i, store tuples (neighbor, dx, dy)
    # meaning: from i's perspective, neighbor is dx, dy away
    adjacency = [[] for _ in range(N+1)]
    
    idx = 2
    for _ in range(M):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        X = int(input_data[idx+2]); Y = int(input_data[idx+3])
        idx += 4
        adjacency[A].append((B, X, Y))
        adjacency[B].append((A, -X, -Y))
    
    # We'll find coordinates by BFS from person 1 (if N >= 1)
    coordinates = [None]*(N+1)  # (x,y) or None
    visited = [False]*(N+1)
    
    if N >= 1:
        # Person 1 is at the origin
        coordinates[1] = (0, 0)
        visited[1] = True
        
        queue = deque([1])
        while queue:
            curr = queue.popleft()
            cx, cy = coordinates[curr]
            for (nxt, dx, dy) in adjacency[curr]:
                if not visited[nxt]:
                    visited[nxt] = True
                    coordinates[nxt] = (cx + dx, cy + dy)
                    queue.append(nxt)
                # The problem guarantees consistency, so no need to check for conflicts
    
    # Output results
    out = []
    for i in range(1, N+1):
        if coordinates[i] is None:
            out.append("undecidable")
        else:
            x, y = coordinates[i]
            out.append(f"{x} {y}")
    
    print("
".join(out))

# Do not remove this call
main()