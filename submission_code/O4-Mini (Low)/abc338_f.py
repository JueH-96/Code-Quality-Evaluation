import sys
def main():
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    INF = 10**30

    N, M = map(int, input().split())
    # initialize distance matrix
    d = [[INF]*N for _ in range(N)]
    for i in range(N):
        d[i][i] = 0
    for _ in range(M):
        u,v,w = map(int, input().split())
        u-=1; v-=1
        d[u][v] = min(d[u][v], w)

    # Floyd–Warshall for all-pairs shortest paths
    for k in range(N):
        dk = d[k]
        for i in range(N):
            di = d[i]
            ik = di[k]
            if ik == INF: continue
            # relax di[j] via k
            for j in range(N):
                nd = ik + dk[j]
                if nd < di[j]:
                    di[j] = nd

    # Check reachability: if there is no way to reach some vertex at all,
    # then no covering walk can exist.
    # We only need that every vertex is reachable from at least one others,
    # but in DP we will see INF and skip.  Still we can do a quick check:
    # If there is a vertex v such that for every u!=v d[u][v] is INF,
    # and v!=u so must start at v; but that is okay.  Actually we don't need
    # a preliminary check: the DP will find INF as the final answer.
    FULL = (1<<N) - 1
    # dp[mask][v] = best cost ending at v having visited exactly mask
    dp = [ [INF]*N for _ in range(1<<N) ]

    # base cases: start at each single vertex
    for v in range(N):
        dp[1<<v][v] = 0

    # iterate all masks
    for mask in range(1<<N):
        # for each end-vertex u in mask
        # if dp[mask][u] is finite, try to go to any v not in mask
        # cost += d[u][v]
        row = dp[mask]
        if all(x==INF for x in row):
            continue
        rem = FULL ^ mask
        # iterate set bits of mask
        ubit = mask
        while ubit:
            u = (ubit & -ubit).bit_length() - 1
            cost_u = row[u]
            ubit &= ubit - 1
            if cost_u == INF:
                continue
            # try extend to each v not yet visited
            r = rem
            du = d[u]
            while r:
                v = (r & -r).bit_length() - 1
                r &= r - 1
                nd = cost_u + du[v]
                if nd < dp[mask | (1<<v)][v]:
                    dp[mask | (1<<v)][v] = nd

    ans = min(dp[FULL])
    if ans >= INF//2:
        print("No")
    else:
        print(ans)

if __name__ == "__main__":
    main()