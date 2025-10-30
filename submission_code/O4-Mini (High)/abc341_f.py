def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    graph = [[] for _ in range(N)]
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        graph[u].append(v)
        graph[v].append(u)

    W = list(map(int, input().split()))
    A = list(map(int, input().split()))

    # Process vertices in increasing order of weight
    nodes = list(range(N))
    nodes.sort(key=lambda i: W[i])

    # f[x] = maximum number of operations starting from a single piece at x
    f = [0] * N

    graphL = graph
    WL = W
    fL = f

    for x in nodes:
        wx = WL[x]
        cap = wx - 1
        best = 0
        if cap > 0:
            # dp[c] = maximum sum of f[y] using neighbors of x with total W <= c
            dp = [0] * (cap + 1)
            dp_local = dp
            neigh = graphL[x]
            fl = fL
            wl = WL
            for y in neigh:
                wy = wl[y]
                if wy < wx:
                    fy = fl[y]
                    # 0-1 knapsack update for this item (weight=wy, value=fy)
                    # iterate capacities from cap down to wy
                    start = wy
                    for c in range(cap, start - 1, -1):
                        new_val = dp_local[c - wy] + fy
                        if new_val > dp_local[c]:
                            dp_local[c] = new_val
                            if new_val > best:
                                best = new_val
        # plus one for removing the piece at x
        fL[x] = best + 1

    # Total operations = sum over all initial pieces A[i] * f[i]
    total = 0
    for i in range(N):
        ai = A[i]
        if ai:
            total += ai * f[i]

    print(total)

# call main to execute
main()