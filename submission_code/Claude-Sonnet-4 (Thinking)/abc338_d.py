N, M = map(int, input().split())
X = list(map(int, input().split()))

min_tour_length = float('inf')

# Try removing each bridge
for removed_bridge in range(1, N+1):
    # Determine the path after removing the bridge
    if removed_bridge == N:
        # Remove edge (N, 1)
        # The path is: 1 - 2 - 3 - ... - N
        path = list(range(1, N+1))
    else:
        # Remove edge (removed_bridge, removed_bridge + 1)
        u = removed_bridge
        # The path goes from u backwards to 1, then from N backwards to u+1
        path = []
        # Start from u and go backwards to 1
        for i in range(u, 0, -1):
            path.append(i)
        # Then from N backwards to u+1
        for i in range(N, u, -1):
            path.append(i)
    
    # Create a mapping from node to position on the path
    node_to_pos = {}
    for pos, node in enumerate(path):
        node_to_pos[node] = pos
    
    # Compute the tour length
    tour_length = 0
    for i in range(M - 1):
        a, b = X[i], X[i + 1]
        tour_length += abs(node_to_pos[a] - node_to_pos[b])
    
    min_tour_length = min(min_tour_length, tour_length)

print(min_tour_length)