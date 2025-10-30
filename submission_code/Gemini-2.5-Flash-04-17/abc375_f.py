import sys

# Use a large value for infinity
# Max possible distance can be up to N * max_edge_weight ≈ 300 * 10^9 = 3 * 10^11.
# This value fits within a signed 64-bit integer.
# sys.maxsize is typically larger than 2^63, suitable for representing infinity.
INF = sys.maxsize

def solve():
    # Read N, M, Q
    N, M, Q = map(int, sys.stdin.readline().split())

    # Store road data. Roads are 1-indexed in input, store as (u-1, v-1, c) (0-indexed cities).
    roads_data = []
    for _ in range(M):
        u, v, c = map(int, sys.stdin.readline().split())
        roads_data.append((u - 1, v - 1, c))

    # Track closed roads status. Initially all roads are open.
    # is_closed[i] corresponds to road i+1 in input.
    is_closed = [False] * M

    # Function to compute All-Pairs Shortest Paths (APSP) using Floyd-Warshall
    # for the current graph state (based on which roads are not closed).
    def compute_apsp():
        # Initialize distance matrix. dist[i][j] will store the shortest distance
        # between city i and city j (0-indexed).
        dist = [[INF] * N for _ in range(N)]

        # Distance from a city to itself is 0.
        for i in range(N):
            dist[i][i] = 0

        # Add currently open roads to the initial distance matrix.
        # For bidirectional roads, update both directions.
        # If multiple open roads exist between the same pair of cities, take the minimum weight.
        for i in range(M):
            if not is_closed[i]:
                u, v, c = roads_data[i]
                dist[u][v] = min(dist[u][v], c)
                dist[v][u] = min(dist[v][u], c)

        # Floyd-Warshall algorithm main loop.
        # k is the intermediate vertex allowed in paths.
        for k in range(N):
            # i is the starting vertex.
            for i in range(N):
                # j is the ending vertex.
                for j in range(N):
                    # If paths from i to k AND k to j exist (i.e., not INF),
                    # check if the path i -> k -> j is shorter than the current shortest path i -> j.
                    # The check `dist[i][k] != INF and dist[k][j] != INF` is important
                    # to prevent issues if INF + INF were to wrap around or behave unexpectedly
                    # depending on the language's integer limits (though Python handles large ints).
                    if dist[i][k] != INF and dist[k][j] != INF:
                         dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        return dist

    # Initially, all roads are open. Compute the first APSP matrix.
    # This result will be used to answer type 2 queries until the first type 1 query occurs.
    # This initial computation takes O(N^3) time.
    current_dist_matrix = compute_apsp()

    # Process queries one by one in the given order.
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        query_type = query[0]

        if query_type == 1:
            # Query Type 1: Close a road.
            # The road index is 1-based in input.
            road_idx = query[1] - 1 # Convert to 0-indexed road index.

            # Update the status of the specified road to closed.
            # The problem guarantees this road is not already closed.
            is_closed[road_idx] = True

            # The graph structure changes due to the road closure.
            # The current_dist_matrix is now outdated.
            # Recompute the APSP matrix for the new graph state (with the road removed).
            # This recomputation takes O(N^3) time.
            # This occurs only when a type 1 query happens, at most 300 times in total.
            current_dist_matrix = compute_apsp()

        elif query_type == 2:
            # Query Type 2: Find shortest distance between two cities.
            # City indices x and y are 1-based in input.
            x, y = query[1] - 1, query[2] - 1 # Convert to 0-indexed city indices.

            # Look up the shortest distance from city x to city y in the current APSP matrix.
            # This lookup takes O(1) time.
            dist = current_dist_matrix[x][y]

            # Output the distance. If the distance is INF (meaning unreachable), print -1.
            if dist == INF:
                print(-1)
            else:
                print(dist)

# Read input from stdin and run the main solve function.
solve()