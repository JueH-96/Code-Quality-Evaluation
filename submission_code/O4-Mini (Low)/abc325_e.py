import sys
import threading

def main():
    import sys
    import heapq

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = int(next(it))
    B = int(next(it))
    C = int(next(it))

    # Read distance matrix D
    D = [ [0]*N for _ in range(N) ]
    for i in range(N):
        for j in range(N):
            D[i][j] = int(next(it))

    # We'll build a 2-layer graph:
    # layer 0 = using car (or not yet switched to train)
    # layer 1 = after switching (train only)
    #
    # States are (city, layer), city in [0..N-1], layer in {0,1}.
    # From (i,0) -> (j,0) with cost D[i][j]*A  (by car)
    # From (i,0) -> (i,1) with cost 0           (switch to train)
    # From (i,1) -> (j,1) with cost D[i][j]*B + C  (by train)
    #
    # We run Dijkstra from (0,0) to reach min of dist[(N-1,0)] or dist[(N-1,1)].

    INF = 10**30
    dist = [ [INF]*2 for _ in range(N) ]
    dist[0][0] = 0

    # min-heap of (current_distance, city, layer)
    hq = [ (0, 0, 0) ]

    visited = [ [False]*2 for _ in range(N) ]
    target = N-1

    while hq:
        curd, u, layer = heapq.heappop(hq)
        if curd > dist[u][layer]:
            continue
        if visited[u][layer]:
            continue
        visited[u][layer] = True

        # Early exit: if we've visited both layers of target, we can stop
        if u == target and visited[target][0] and visited[target][1]:
            break

        if layer == 0:
            # 1) travel by car to any other city in layer 0
            # 2) switch to train at same city
            # (2) switching:
            if dist[u][1] > curd:
                dist[u][1] = curd
                heapq.heappush(hq, (curd, u, 1))
            # (1) drive edges
            du = D[u]
            for v in range(N):
                if v == u:
                    continue
                nd = curd + du[v]*A
                if nd < dist[v][0]:
                    dist[v][0] = nd
                    heapq.heappush(hq, (nd, v, 0))

        else:
            # layer == 1: can only move by train
            du = D[u]
            for v in range(N):
                if v == u:
                    continue
                nd = curd + du[v]*B + C
                if nd < dist[v][1]:
                    dist[v][1] = nd
                    heapq.heappush(hq, (nd, v, 1))

    # Answer is min distance to target in either layer
    ans = min(dist[target][0], dist[target][1])
    print(ans)

if __name__ == "__main__":
    main()