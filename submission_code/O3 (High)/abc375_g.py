import sys
import heapq


def dijkstra_with_counts(start, n, adj, MODS):
    INF = 10 ** 19
    dist = [INF] * (n + 1)
    cnt = [[0] * (n + 1) for _ in MODS]

    dist[start] = 0
    for i, mod in enumerate(MODS):
        cnt[i][start] = 1

    pq = [(0, start)]
    while pq:
        d, v = heapq.heappop(pq)
        if d != dist[v]:
            continue
        for nv, w in adj[v]:
            nd = d + w
            if nd < dist[nv]:
                dist[nv] = nd
                heapq.heappush(pq, (nd, nv))
                for i, mod in enumerate(MODS):
                    cnt[i][nv] = cnt[i][v]
            elif nd == dist[nv]:
                for i, mod in enumerate(MODS):
                    cnt[i][nv] = (cnt[i][nv] + cnt[i][v]) % mod
    return dist, cnt


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))

    edges = []
    adj = [[] for _ in range(N + 1)]
    for idx in range(M):
        a = int(next(it))
        b = int(next(it))
        c = int(next(it))
        edges.append((a, b, c))
        adj[a].append((b, c))
        adj[b].append((a, c))

    MODS = (1_000_000_007, 1_000_000_009, 998_244_353)

    dist1, cnt1 = dijkstra_with_counts(1, N, adj, MODS)
    distN, cntN = dijkstra_with_counts(N, N, adj, MODS)

    D = dist1[N]  # shortest distance with all roads

    total_paths_mod = [cnt1[i][N] for i in range(len(MODS))]

    out_lines = []
    for a, b, c in edges:
        used = False
        prod_equal = True

        if dist1[a] + c + distN[b] == D:
            used = True
            for i, mod in enumerate(MODS):
                prod = (cnt1[i][a] * cntN[i][b]) % mod
                if prod != total_paths_mod[i]:
                    prod_equal = False
                    break
        elif dist1[b] + c + distN[a] == D:
            used = True
            for i, mod in enumerate(MODS):
                prod = (cnt1[i][b] * cntN[i][a]) % mod
                if prod != total_paths_mod[i]:
                    prod_equal = False
                    break
        else:
            prod_equal = False  # edge not on any shortest path

        out_lines.append("Yes" if used and prod_equal else "No")

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()