import sys

def solve():
    """
    Solves the problem by processing queries offline and in reverse.
    The core idea is to start with the final graph (after all road closures)
    and add roads back one by one, updating all-pairs shortest paths at each step.
    """
    # Use fast I/O
    input = sys.stdin.readline

    try:
        line = input()
        if not line: return
        N, M, Q = map(int, line.split())
    except (IOError, ValueError):
        return

    # Store all roads with 0-based indexing for nodes
    roads = []
    for _ in range(M):
        u, v, c = map(int, input().split())
        roads.append((u - 1, v - 1, c))

    # Store all queries with 0-based indexing for nodes/roads
    queries = []
    for _ in range(Q):
        query_line = list(map(int, input().split()))
        if query_line[0] == 1:
            queries.append(('close', query_line[1] - 1))
        else:
            queries.append(('path', query_line[1] - 1, query_line[2] - 1))

    # Identify all roads that will be closed at any point
    closed_road_indices = set()
    for q_type, val1, *_ in queries:
        if q_type == 'close':
            closed_road_indices.add(val1)

    # Initialize the distance matrix for the final state of the graph
    # (i.e., with all 'closed' roads removed).
    INF = float('inf')
    dist = [[INF] * N for _ in range(N)]

    for i in range(N):
        dist[i][i] = 0

    for i in range(M):
        if i not in closed_road_indices:
            u, v, w = roads[i]
            dist[u][v] = min(dist[u][v], w)
            dist[v][u] = min(dist[v][u], w)

    # Compute all-pairs shortest paths for the initial graph using Floyd-Warshall
    for k in range(N):
        for i in range(N):
            for j in range(N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Process queries in reverse order
    results = []
    for query in reversed(queries):
        q_type, val1, *rest = query
        
        if q_type == 'path':
            x, y = val1, rest[0]
            d = dist[x][y]
            if d == INF:
                results.append(-1)
            else:
                results.append(int(d))
        
        elif q_type == 'close':  # In reverse, this means opening a road
            road_idx = val1
            u, v, w = roads[road_idx]
            
            # To update all-pairs shortest paths after adding an edge (u, v, w),
            # we check for every pair of nodes (i, j) if a path via the new
            # edge is shorter. A copy of the distance matrix is used to ensure
            # that the update calculations are based on the state before this
            # edge was added, preventing incorrect paths that might use the
            # new edge multiple times.
            old_dist = [row[:] for row in dist]
            for i in range(N):
                for j in range(N):
                    # Candidate path via u -> v
                    path_uv = old_dist[i][u] + w + old_dist[v][j]
                    # Candidate path via v -> u
                    path_vu = old_dist[i][v] + w + old_dist[u][j]
                    
                    dist[i][j] = min(old_dist[i][j], path_uv, path_vu)
    
    # Print the collected results in the original query order
    for ans in reversed(results):
        print(ans)

# Run the solution
solve()