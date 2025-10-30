import sys

def solve():
    """
    Solves the Guarded Vertices problem using a linear-time, bucket-based
    Dijkstra-like algorithm (Dial's algorithm).
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem size parameters
    try:
        line = input()
        if not line: return
        N, M, K = map(int, line.split())
    except (IOError, ValueError):
        return

    # Build the graph using an adjacency list
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    # Handle the edge case of no guards
    if K == 0:
        print(0)
        print()
        return

    # Read all guard information to determine the maximum stamina
    guards_info = [tuple(map(int, input().split())) for _ in range(K)]
    
    max_h_val = 0
    if guards_info:
        max_h_val = max(h for _, h in guards_info)

    # max_reach[v] stores the maximum remaining stamina at vertex v
    max_reach = [-1] * (N + 1)
    
    # Buckets for each stamina level, for Dial's algorithm
    buckets = [[] for _ in range(max_h_val + 1)]

    # Initialize with guard locations. Since all p_i are distinct,
    # we can initialize directly. The algorithm will handle cases where a guard's
    # location is better protected by another nearby guard.
    for p, h in guards_info:
        max_reach[p] = h
        buckets[h].append(p)
    
    # Process vertices in order of decreasing stamina
    for h in range(max_h_val, 0, -1):
        for u in buckets[h]:
            # If a better path to u has been found, this entry is stale.
            if h < max_reach[u]:
                continue
            
            # Propagate to neighbors
            for v in adj[u]:
                next_h = h - 1
                # If this path provides higher stamina to the neighbor, update it.
                if next_h > max_reach[v]:
                    max_reach[v] = next_h
                    buckets[next_h].append(v)
    
    # Collect all guarded vertices (those with non-negative remaining stamina)
    guarded_vertices = [i for i in range(1, N + 1) if max_reach[i] >= 0]
    
    # Print the results in the specified format
    print(len(guarded_vertices))
    print(*guarded_vertices)

solve()