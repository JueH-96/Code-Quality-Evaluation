# YOUR CODE HERE
import sys

import threading

def main():
    import sys
    import math
    import heapq

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M, Q = map(int, sys.stdin.readline().split())
    NODES = N
    EDGES = M
    QUERIES = Q

    INF = float('inf')

    w = [[INF]*N for _ in range(N)]

    for i in range(N):
        w[i][i] = 0

    for i in range(M):
        A,B,C = map(int, sys.stdin.readline().split())
        A -=1
        B -=1
        w[A][B] = min(w[A][B], C)
        w[B][A] = min(w[B][A], C)

    # Prepare list of queries
    queries = []
    edge_closure_times = []
    edge_delete_queries = []
    for idx in range(Q):
        tokens = sys.stdin.readline().split()
        if tokens[0] == '1':
            edge_delete_queries.append((idx, int(tokens[1])-1))
            queries.append(('1', int(tokens[1])-1))
        else:
            x = int(tokens[1])-1
            y = int(tokens[2])-1
            queries.append(('2', x, y))

    # Build edge closure mapping
    edge_closure_time = [Q]*M  # Initialize closure time to after all queries
    for (time, edge_idx) in edge_delete_queries:
        edge_closure_time[edge_idx] = time

    # Build time intervals between edge deletions
    time_points = sorted(set([0] + [t for t, _ in edge_delete_queries] + [Q]))
    intervals = []
    for i in range(len(time_points)-1):
        intervals.append((time_points[i], time_points[i+1]))

    # Prepare adjacency matrices per interval
    # Since we're removing edges, need to prepare graphs per interval
    # For each interval, edges that are still open are those with closure time > interval_start

    # Precompute the edges that are available in each interval
    # Edges are (A_i, B_i, C_i, edge_idx)

    edges = []
    for idx in range(M):
        edges.append((idx, edge_closure_time[idx]))

    # Now process each interval
    D = [[INF]*N for _ in range(N)]
    for i in range(N):
        D[i][i] = 0

    # Initialize adjacency matrix
    w_current = [row[:] for row in w]

    interval_idx = 0
    edge_idx_set = set(range(M))

    query_idx = 0
    while query_idx < Q:
        query = queries[query_idx]
        # Process queries until next edge deletion
        if query[0] == '1':
            # Edge deletion
            # Recompute D[i][j] after edge deletion
            # Remove edge
            edge_idx = query[1]
            A_i = None
            B_i = None
            for row_idx in range(N):
                for col_idx in range(N):
                    if w[row_idx][col_idx] == w_current[row_idx][col_idx]:
                        continue
                    w_current[row_idx][col_idx] = w[row_idx][col_idx]
            # Remove the edge
            for row_idx in range(N):
                for col_idx in range(N):
                    w_current[row_idx][col_idx] = w[row_idx][col_idx]

            for i in range(N):
                for j in range(N):
                    D[i][j] = w_current[i][j]
            # Run Floyd-Warshall algorithm to recompute distances
            for k in range(N):
                for i in range(N):
                    for j in range(N):
                        if D[i][k] + D[k][j] < D[i][j]:
                            D[i][j] = D[i][k] + D[k][j]
            # Remove edge from w_current
            for i in range(N):
                w_current[i][i] = 0
            idx = edge_idx
            # Get edge info
            a = None
            b = None
            c = None
            edge_line = idx
            count = 0
            with open(sys.argv[0], 'r') as f:
                for line in f:
                    if count == edge_line + 1:
                        tokens = line.split()
                        a = int(tokens[0])-1
                        b = int(tokens[1])-1
                        c = int(tokens[2])
                        break
                    count +=1
            # Remove edge
            w_current[a][b] = INF
            w_current[b][a] = INF
            # Recompute distances after edge removal
            for i in range(N):
                for j in range(N):
                    D[i][j] = w_current[i][j]
            for k in range(N):
                for i in range(N):
                    for j in range(N):
                        if D[i][k] + D[k][j] < D[i][j]:
                            D[i][j] = D[i][k] + D[k][j]
            query_idx +=1
        else:
            # Distance query
            x = query[1]
            y = query[2]
            dist = D[x][y]
            if dist == INF:
                print(-1)
            else:
                print(int(dist))
            query_idx +=1

# Run the main function in a separate thread
threading.Thread(target=main).start()