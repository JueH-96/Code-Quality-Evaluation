def solve():
    import sys
    from heapq import heappush, heappop

    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    edges_data = input_data[3:3+2*M]
    guards_data = input_data[3+2*M:]

    # Build adjacency list
    adj = [[] for _ in range(N)]
    idx = 0
    for _ in range(M):
        a = int(edges_data[idx]) - 1
        b = int(edges_data[idx+1]) - 1
        idx += 2
        adj[a].append(b)
        adj[b].append(a)

    # Read guard positions and stamina
    guards = []
    for i in range(K):
        p_i = int(guards_data[2*i]) - 1
        h_i = int(guards_data[2*i+1])
        guards.append((p_i, h_i))

    # dist[v] will hold the maximum "stamina" available to reach v, or -1 if unreachable
    dist = [-1]*N

    # Max-heap for BFS (store as negative for use with Python's min-heap)
    heap = []
    for p_i, h_i in guards:
        dist[p_i] = h_i
        heappush(heap, (-h_i, p_i))

    # BFS-like process with a priority queue
    while heap:
        cur_stamina, node = heappop(heap)
        cur_stamina = -cur_stamina  # Convert back to positive

        # If this node's dist is better than the stamina we popped, skip
        if dist[node] != cur_stamina:
            continue

        # Spread to neighbors if we still have stamina left
        if cur_stamina > 0:
            for nxt in adj[node]:
                if dist[nxt] < cur_stamina - 1:
                    dist[nxt] = cur_stamina - 1
                    heappush(heap, (-(cur_stamina - 1), nxt))

    # Gather guarded vertices (dist[v] >= 0)
    guarded_vertices = [i+1 for i in range(N) if dist[i] >= 0]
    print(len(guarded_vertices))
    print(" ".join(map(str, guarded_vertices)))