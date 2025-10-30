import sys
import heapq

def main():
    import sys
    import heapq
    from math import inf

    # Read all input at once
    data = sys.stdin.read().split()
    idx = 0

    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1

    # Read vertex weights
    A = [0] * (N + 1)
    for i in range(1, N + 1):
        A[i] = int(data[idx])
        idx += 1

    # Build adjacency list with adjusted edge weights
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        U = int(data[idx])
        idx += 1
        V = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        # Adjusted weight for edge U->V is B + A[V]
        adj[U].append((V, B + A[V]))
        adj[V].append((U, B + A[U]))

    # Dijkstra's algorithm
    dist = [inf] * (N + 1)
    dist[1] = 0  # Starting point

    visited = [False] * (N + 1)
    pq = [(0, 1)]  # (current distance, vertex)

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        for v, weight in adj[u]:
            if not visited[v]:
                new_dist = dist[u] + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

    # Collect results for vertices 2 to N
    results = [str(dist[i] + A[1]) for i in range(2, N + 1)]
    print(' '.join(results))

if __name__ == "__main__":
    main()