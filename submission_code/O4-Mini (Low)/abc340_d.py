import sys
import threading

def main():
    import sys
    import heapq

    input = sys.stdin.readline
    N = int(input().strip())
    # Build adjacency list: graph[u] = list of (v, cost)
    graph = [[] for _ in range(N+1)]
    for i in range(1, N):
        A, B, X = map(int, input().split())
        # edge i -> i+1 with cost A
        graph[i].append((i+1, A))
        # edge i -> X with cost B
        graph[i].append((X, B))

    # Dijkstra from 1 to N
    INF = 10**30
    dist = [INF] * (N+1)
    dist[1] = 0
    hq = [(0, 1)]  # (distance, node)

    while hq:
        d, u = heapq.heappop(hq)
        if d > dist[u]:
            continue
        if u == N:
            # We can early exit if we reached N with minimal dist
            break
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(hq, (nd, v))

    print(dist[N])

if __name__ == "__main__":
    main()