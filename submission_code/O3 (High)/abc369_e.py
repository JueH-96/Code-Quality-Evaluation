import sys
from itertools import permutations

INF = 10 ** 20          # larger than every possible answer


def floyd_warshall(dist):
    """all–pairs shortest paths, 0-based vertices"""
    n = len(dist)
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:          # unreachable – skip
                continue
            di = dist[i]
            base = dik
            for j in range(n):
                tmp = base + dk[j]
                if tmp < di[j]:
                    di[j] = tmp


def main() -> None:
    rd = sys.stdin.buffer.read().split()
    it = iter(rd)

    # ------------ read graph -----------------
    n = int(next(it))          # islands
    m = int(next(it))          # bridges

    # store every bridge for later queries
    u = [0] * (m + 1)
    v = [0] * (m + 1)
    t = [0] * (m + 1)

    # distance matrix (only min weight kept for parallel edges)
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0

    for idx in range(1, m + 1):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        w = int(next(it))
        u[idx], v[idx], t[idx] = a, b, w
        if w < dist[a][b]:
            dist[a][b] = dist[b][a] = w

    # ------------- preprocessing --------------
    floyd_warshall(dist)

    start = 0         # island 1  (0-based)
    goal = n - 1      # island N

    # ------------- answer queries -------------
    q = int(next(it))
    out_lines = []

    for _ in range(q):
        k = int(next(it))              # number of mandatory bridges ( ≤ 5 )
        must = []
        for _j in range(k):
            idx = int(next(it))
            must.append((u[idx], v[idx], t[idx]))   # (end1, end2, weight)

        best = INF
        idx_range = range(k)

        for order in permutations(idx_range):       # ≤ 120 permutations
            for mask in range(1 << k):              # ≤ 32 orientation patterns
                # ----- first mandatory bridge -----
                i0 = order[0]
                a, b, w = must[i0]
                if (mask >> i0) & 1:                # reverse orientation ?
                    a, b = b, a

                cost = dist[start][a] + w
                if cost >= best:
                    continue
                cur = b

                # ----- remaining mandatory bridges -----
                for i in order[1:]:
                    a, b, w = must[i]
                    if (mask >> i) & 1:
                        a, b = b, a
                    cost += dist[cur][a] + w
                    if cost >= best:
                        break
                    cur = b
                else:
                    cost += dist[cur][goal]
                    if cost < best:
                        best = cost

        out_lines.append(str(best))

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()