import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin
    line = data.readline().split()
    if not line:
        print(-1)
        return
    N = int(line[0]); Q = int(line[1])
    intervals = []
    sumC = 0
    for _ in range(Q):
        lrc = data.readline().split()
        if not lrc:
            lrc = data.readline().split()
        l, r, c = map(int, lrc)
        intervals.append((c, l, r))
        sumC += c
    # We need to cover adjacent pairs j, j+1 for j in [1..N-1]
    # cost[j] = minimal C among intervals covering [j, j+1].
    # If any cost[j] remains INF, graph disconnected.
    INF = 10**30
    cost = [INF] * (N + 1)  # we'll use cost[1..N-1]
    # DSU for "next free" pointer
    parent = list(range(N+2))  # parent[i] = next free position >= i

    def find(x):
        # path-compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # sort intervals by cost ascending
    intervals.sort()
    for c, l, r in intervals:
        # we need to assign cost[j]=c for all j in [l, r-1] not yet assigned
        j = find(l)
        # we iterate while j <= r-1
        end = r - 1
        while j <= end:
            # assign
            cost[j] = c
            # mark j as processed: skip to j+1
            parent[j] = j + 1
            # move to next free
            j = find(j)
    # check coverage
    if N >= 2:
        for j in range(1, N):
            if cost[j] >= INF:
                print(-1)
                return
    # sum cost for adjacent edges
    tot_adj = 0
    # range 1..N-1
    for j in range(1, N):
        tot_adj += cost[j]
    ans = tot_adj + sumC
    print(ans)

if __name__ == "__main__":
    main()