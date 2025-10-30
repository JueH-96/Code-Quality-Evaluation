import sys
import threading

def main():
    import sys
    data = sys.stdin
    N = int(data.readline().strip())
    # Build adjacency list for a directed graph with N nodes (1-indexed).
    # For each stage i (1 <= i < N):
    #  - edge i -> i+1 with cost A_i
    #  - edge i -> X_i  with cost B_i
    graph = [[] for _ in range(N+1)]
    for i in range(1, N):
        A, B, X = map(int, data.readline().split())
        graph[i].append((i+1, A))
        graph[i].append((X,   B))
    # We don't add edges from stage N since there is no A_N, B_N, X_N.

    # Dijkstra's algorithm from node 1 to node N
    import heapq
    INF = 10**30
    dist = [INF] * (N+1)
    dist[1] = 0
    hq = [(0, 1)]  # (current_distance, node)

    while hq:
        d, u = heapq.heappop(hq)
        if d > dist[u]:
            continue
        if u == N:
            # Once we pop N with its shortest distance we can stop early
            break
        for v, w in graph[u]:
            nd = d + w
            # Relaxation step
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))

    # The answer is the shortest distance to node N
    print(dist[N])

if __name__ == "__main__":
    main()