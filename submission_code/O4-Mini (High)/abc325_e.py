import sys
import threading

def main():
    import sys
    import heapq

    input = sys.stdin.readline
    INF = 10**30

    # Read inputs
    N, A, B, C = map(int, input().split())
    D = [list(map(int, input().split())) for _ in range(N)]

    # Dijkstra on the "car" graph from city 0
    def dijkstra_car():
        dist = [INF] * N
        dist[0] = 0
        hq = [(0, 0)]
        heappush = heapq.heappush
        heappop = heapq.heappop
        Dlocal = D
        Alocal = A
        Nlocal = N

        while hq:
            d, u = heappop(hq)
            if d != dist[u]:
                continue
            Du = Dlocal[u]
            # relax all edges u->v in car graph
            for v in range(Nlocal):
                nd = d + Du[v] * Alocal
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(hq, (nd, v))
        return dist

    # Dijkstra on the "train" graph from city N-1
    def dijkstra_train():
        dist = [INF] * N
        dist[N-1] = 0
        hq = [(0, N-1)]
        heappush = heapq.heappush
        heappop = heapq.heappop
        Dlocal = D
        Blocal = B
        Clocal = C
        Nlocal = N

        while hq:
            d, u = heappop(hq)
            if d != dist[u]:
                continue
            Du = Dlocal[u]
            # relax all edges u->v in train graph
            for v in range(Nlocal):
                nd = d + Du[v] * Blocal + Clocal
                if nd < dist[v]:
                    dist[v] = nd
                    heappush(hq, (nd, v))
        return dist

    dist_car   = dijkstra_car()
    dist_train = dijkstra_train()

    # Combine: choose switching city k to minimize dist_car[k] + dist_train[k]
    ans = INF
    for k in range(N):
        t = dist_car[k] + dist_train[k]
        if t < ans:
            ans = t

    print(ans)

if __name__ == "__main__":
    main()