def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))

    # Build graph (1-indexed)
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)

    # dp[v] will record the maximum stamina remaining available from any guard
    # that can reach vertex v. If dp[v] is non-negative, then vertex v is guarded.
    dp = [-1] * (N + 1)
    dq = deque()

    # Initialize the multi-source BFS with guard positions
    for _ in range(K):
        p = int(next(it))
        h = int(next(it))
        dp[p] = h
        dq.append(p)

    # Perform multi-source BFS: Propagate from each guard
    while dq:
        current = dq.popleft()
        # If we don't have any stamina left to spread, skip neighbors.
        if dp[current] <= 0:
            continue
        # Propagate to each neighbor.
        for neighbor in graph[current]:
            new_stamina = dp[current] - 1
            if new_stamina > dp[neighbor]:
                dp[neighbor] = new_stamina
                dq.append(neighbor)

    # Collect all guarded vertices. A vertex is guarded if dp[v] >= 0.
    guarded_vertices = [str(v) for v in range(1, N + 1) if dp[v] >= 0]
    guarded_vertices.sort(key=int)

    result = []
    result.append(str(len(guarded_vertices)))
    if guarded_vertices:
        result.append(" ".join(guarded_vertices))
    sys.stdout.write("
".join(result))


if __name__ == "__main__":
    main()