import sys

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = int(next(it))
    B = int(next(it))
    C = int(next(it))
    # Read distance matrix
    D = [ [0]*N for _ in range(N) ]
    for i in range(N):
        for j in range(N):
            D[i][j] = int(next(it))

    # A large infinity value for distances
    INF = 10**30

    # Dijkstra in O(N^2) for complete graph
    def dijkstra(src, mode):
        # mode = 0 for car, 1 for train
        dist = [INF] * N
        used = [False] * N
        dist[src] = 0
        for _ in range(N):
            # pick unvisited node with smallest dist
            u = -1
            best = INF
            for i in range(N):
                if not used[i] and dist[i] < best:
                    best = dist[i]
                    u = i
            if u == -1:
                break
            used[u] = True
            du = dist[u]
            if mode == 0:
                # relax car edges
                mul = A
                rowD = D[u]
                for v in range(N):
                    if not used[v]:
                        alt = du + mul * rowD[v]
                        if alt < dist[v]:
                            dist[v] = alt
            else:
                # relax train edges
                mul = B
                rowD = D[u]
                add = C
                for v in range(N):
                    if not used[v]:
                        alt = du + mul * rowD[v] + add
                        if alt < dist[v]:
                            dist[v] = alt
        return dist

    # dist_car[i]: min time from city 1 (index 0) to i by car-only
    dist_car = dijkstra(0, 0)
    # dist_train[i]: min time from city N (index N-1) to i by train-only
    dist_train = dijkstra(N-1, 1)

    # Answer: min over switch city s: dist_car[s] + dist_train[s]
    ans = INF
    for i in range(N):
        t = dist_car[i] + dist_train[i]
        if t < ans:
            ans = t

    # Print result
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()