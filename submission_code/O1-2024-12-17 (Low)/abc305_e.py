def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    edges_part = input_data[3:3 + 2*M]
    guards_part = input_data[3 + 2*M:]

    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(M):
        a = int(edges_part[idx])
        b = int(edges_part[idx+1])
        idx += 2
        adj[a].append(b)
        adj[b].append(a)

    # Read guards
    # p_i, h_i
    guards = []
    for i in range(K):
        p = int(guards_part[2*i])
        h = int(guards_part[2*i+1])
        guards.append((p, h))

    # best[node] = maximum stamina that can cover that node
    best = [-1]*(N+1)

    # max-heap (we use min-heap with negative values in Python)
    heap = []
    for p, h in guards:
        best[p] = h
        # push (h, p) as (-h, p) to simulate max-heap
        heapq.heappush(heap, (-h, p))

    # Multi-source "BFS" using a priority queue
    while heap:
        negd, u = heapq.heappop(heap)
        d = -negd  # recover the positive stamina
        # If we do not match the best stamina known, skip
        if best[u] != d:
            continue
        # Propagate to neighbors if we can go further
        if d > 0:
            for v in adj[u]:
                if best[v] < d - 1:
                    best[v] = d - 1
                    heapq.heappush(heap, (-(d - 1), v))

    # Collect guarded vertices
    guarded = [i for i in range(1, N+1) if best[i] >= 0]
    print(len(guarded))
    print(*guarded)


# Do not forget to call main()
if __name__ == "__main__":
    main()