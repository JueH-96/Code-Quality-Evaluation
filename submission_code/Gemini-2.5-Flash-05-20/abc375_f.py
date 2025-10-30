import sys

def solve():
    # Read N, M, Q
    N, M, Q = map(int, sys.stdin.readline().split())

    # Store original road information. 
    # Cities are 1-indexed in input, convert to 0-indexed for array access.
    # roads_info[i] = (u, v, cost) for the i-th road (0-indexed)
    roads_info = []
    for _ in range(M):
        u, v, c = map(int, sys.stdin.readline().split())
        roads_info.append((u - 1, v - 1, c))

    # road_status[i] is True if road i (0-indexed) is currently closed.
    road_status = [False] * M

    # Use float('inf') for unreachable paths
    INF = float('inf')

    # Adjacency matrix to store current shortest distances between all pairs of cities.
    # This matrix will be updated after each road closure.
    dist = [[INF] * N for _ in range(N)]

    # Function to recompute All-Pairs Shortest Paths using Floyd-Warshall
    def recompute_apsp():
        # Initialize the 'dist' matrix for the current graph state:
        # Set dist[i][j] to the direct edge cost if it's open, else INF.
        # Set dist[i][i] to 0.
        for i in range(N):
            for j in range(N):
                if i == j:
                    dist[i][j] = 0
                else:
                    dist[i][j] = INF

        # Add currently open direct edges to the dist matrix
        for i in range(M):
            if not road_status[i]: # If road i is NOT closed
                u, v, c = roads_info[i]
                # Roads are bidirectional, so update both directions
                dist[u][v] = min(dist[u][v], c)
                dist[v][u] = min(dist[v][u], c)

        # Run Floyd-Warshall algorithm
        # k is the intermediate node
        for k in range(N):
            # i is the source node
            for i in range(N):
                # j is the destination node
                for j in range(N):
                    # If path from i to k and k to j exist
                    if dist[i][k] != INF and dist[k][j] != INF:
                        # Update shortest path from i to j via k
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Initial APSP computation before any queries are processed
    recompute_apsp()

    results = [] # Store results for type 2 queries

    # Process all queries
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        query_type = query[0]

        if query_type == 1:
            # Type 1 query: Close road 'i'
            road_idx = query[1] - 1 # Convert 1-indexed road to 0-indexed
            road_status[road_idx] = True # Mark road as closed
            recompute_apsp() # Recompute APSP after the road closure
        elif query_type == 2:
            # Type 2 query: Find shortest distance from x to y
            x, y = query[1] - 1, query[2] - 1 # Convert 1-indexed cities to 0-indexed
            shortest_dist = dist[x][y] # Lookup the precomputed shortest distance
            if shortest_dist == INF:
                results.append("-1") # If unreachable, print -1
            else:
                results.append(str(shortest_dist)) # Otherwise, print the distance
    
    # Print all results separated by newlines
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to run the program
solve()