def main():
    import sys
    from collections import deque

    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    edges_idx = 2

    # Build adjacency list and degree array
    adj = [[] for _ in range(N + 1)]
    deg = [0] * (N + 1)
    for _ in range(N - 1):
        a, b = map(int, input_data[edges_idx:edges_idx + 2])
        edges_idx += 2
        adj[a].append(b)
        adj[b].append(a)
        deg[a] += 1
        deg[b] += 1

    # Read the required vertices
    required_vertices = set(map(int, input_data[edges_idx:edges_idx + K]))

    # We'll prune leaves that are not in the required set
    removed = [False] * (N + 1)
    queue = deque()

    # Initialize queue with leaves that are NOT required
    for v in range(1, N + 1):
        if deg[v] == 1 and v not in required_vertices:
            queue.append(v)

    # BFS-style pruning
    while queue:
        leaf = queue.popleft()
        if removed[leaf]:
            continue
        removed[leaf] = True
        for neighbor in adj[leaf]:
            deg[neighbor] -= 1
            # If neighbor becomes a leaf, not removed, and not required, enqueue it
            if deg[neighbor] == 1 and not removed[neighbor] and neighbor not in required_vertices:
                queue.append(neighbor)

    # Count the vertices that are not removed
    answer = sum(1 for v in range(1, N + 1) if not removed[v])
    print(answer)

# Don't forget to call main()!
if __name__ == "__main__":
    main()