import sys
import numpy as np

def main():
    # Read all input at once
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1

    # Initialize distance matrix with infinities
    INF = 1e18
    dist = np.full((N, N), INF, dtype=np.float64)
    np.fill_diagonal(dist, 0)

    # Read M roads
    roads = []
    for _ in range(M):
        A = int(data[ptr]) - 1
        ptr += 1
        B = int(data[ptr]) - 1
        ptr += 1
        C = int(data[ptr])
        ptr += 1
        roads.append((A, B, C))
        dist[A, B] = min(dist[A, B], C)
        dist[B, A] = min(dist[B, A], C)

    # Precompute initial all pairs shortest paths
    def floyd_warshall(dist_matrix):
        for k in range(N):
            via_k = dist_matrix[:, k:k+1] + dist_matrix[k:k+1, :]
            dist_matrix = np.minimum(dist_matrix, via_k)
        return dist_matrix

    dist = floyd_warshall(dist.copy())

    # Read all Q queries
    queries = []
    for _ in range(Q):
        query_type = int(data[ptr])
        ptr += 1
        if query_type == 1:
            # Close road i
            road_idx = int(data[ptr]) - 1
            ptr += 1
            A, B, C = roads[road_idx]
            dist[A, B] = INF
            dist[B, A] = INF
            dist = floyd_warshall(dist.copy())
        elif query_type == 2:
            # Query shortest path from x to y
            x = int(data[ptr]) - 1
            ptr += 1
            y = int(data[ptr]) - 1
            ptr += 1
            if dist[x, y] < INF:
                queries.append(int(dist[x, y]))
            else:
                queries.append(-1)
    
    # Print all answers
    print("
".join(map(str, queries)))

if __name__ == '__main__':
    main()