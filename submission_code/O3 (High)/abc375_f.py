import sys


def main() -> None:
    # fast whole-file reading -----------------------------------------------
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    # basic parameters
    N = int(next(it))          # cities
    M = int(next(it))          # roads
    Q = int(next(it))          # queries

    # roads ------------------------------------------------------------------
    A = [0] * M
    B = [0] * M
    C = [0] * M
    for i in range(M):
        A[i] = int(next(it)) - 1      # 0-based
        B[i] = int(next(it)) - 1
        C[i] = int(next(it))

    # read queries, remember which roads get closed --------------------------
    queries = []
    removed = [False] * M          # will be True if the road is closed at least once

    for _ in range(Q):
        t = next(it)
        if t == b'1':              # close road
            idx = int(next(it)) - 1
            queries.append((1, idx))
            removed[idx] = True
        else:                      # distance query
            x = int(next(it)) - 1
            y = int(next(it)) - 1
            queries.append((2, x, y))

    # ------------------------------------------------------------------------
    INF = 10 ** 18                # big enough ( > maximal possible distance )
    dist = [[INF] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0

    # edges that are still open in the final graph
    for i in range(M):
        if not removed[i]:
            u, v, w = A[i], B[i], C[i]
            if w < dist[u][v]:
                dist[u][v] = dist[v][u] = w

    # Floyd-Warshall once on the final graph ---------------------------------
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(N):
                nj = dik + dk[j]
                if nj < di[j]:
                    di[j] = nj

    # helper: add a road and update all-pairs distances in O(N²) -------------
    def add_edge(u: int, v: int, w: int) -> None:
        # if the direct edge itself is shorter, record it first
        if w < dist[u][v]:
            dist[u][v] = dist[v][u] = w

        for i in range(N):
            di = dist[i]
            du = di[u]
            dv = di[v]
            for j in range(N):
                # path i -> u -> v -> j
                val = du + w + dist[v][j]
                if val < di[j]:
                    di[j] = val
                    dist[j][i] = val        # keep symmetry

                # path i -> v -> u -> j
                val2 = dv + w + dist[u][j]
                if val2 < di[j]:
                    di[j] = val2
                    dist[j][i] = val2       # keep symmetry

    # process queries in reverse order ---------------------------------------
    answers = []
    for q in reversed(queries):
        if q[0] == 2:                       # distance query
            x, y = q[1], q[2]
            d = dist[x][y]
            answers.append(str(d if d < INF else -1))
        else:                               # road re-opens (edge addition)
            idx = q[1]
            add_edge(A[idx], B[idx], C[idx])

    # output answers in the original order -----------------------------------
    sys.stdout.write('
'.join(reversed(answers)))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()