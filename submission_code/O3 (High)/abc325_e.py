import sys


def dijkstra(N, weight_mul, add_const, start, dist_matrix):
    """
    Dijkstra for this special complete graph.
    For a move i -> j  the cost is  dist_matrix[i][j] * weight_mul + add_const
    """
    INF = 10 ** 30
    dist = [INF] * N
    dist[start] = 0
    used = [False] * N

    for _ in range(N):
        # pick the unused vertex having minimum tentative distance
        v = -1
        best = INF
        for i in range(N):
            if not used[i] and dist[i] < best:
                best = dist[i]
                v = i
        if v == -1:       # all remaining vertices are unreachable
            break

        used[v] = True
        dv = dist[v]
        row = dist_matrix[v]
        wm = weight_mul
        ac = add_const
        for u in range(N):
            if used[u]:
                continue
            nd = dv + row[u] * wm + ac
            if nd < dist[u]:
                dist[u] = nd
    return dist


def main() -> None:
    data = sys.stdin.read().strip().split()
    it = iter(data)

    N = int(next(it))
    A = int(next(it))
    B = int(next(it))
    C = int(next(it))

    # read distance matrix
    D = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            D[i][j] = int(next(it))

    # car: weight = A * D  (add_const = 0)
    dist_car = dijkstra(N, A, 0, 0, D)

    # train: weight = B * D + C
    # run from city N-1 (index N-1) so we will get distance TO N-1
    dist_train = dijkstra(N, B, C, N - 1, D)

    ans = min(dist_car[i] + dist_train[i] for i in range(N))
    print(ans)


if __name__ == "__main__":
    main()