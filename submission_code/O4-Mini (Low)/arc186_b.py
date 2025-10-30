import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    MOD = 998244353
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    # build intervals [L_i, R_i] for i=1..N
    intervals = []
    for i in range(1, N+1):
        L = A[i-1] + 1
        R = i
        intervals.append((L, R, i))
    # add dummy root as id=0, interval [1,N]
    intervals.append((1, N, 0))
    # sort by increasing L, and decreasing R so that bigger intervals come before nested
    intervals.sort(key=lambda x: (x[0], -x[1]))
    # build tree by stack: parent is the smallest earlier interval that contains it
    children = [[] for _ in range(N+1)]
    stack = []
    for L, R, idx in intervals:
        # pop until stack top contains this interval
        while stack and not (stack[-1][0] <= L and R <= stack[-1][1]):
            stack.pop()
        # now stack[-1] is parent (if exists)
        if stack:
            pidx = stack[-1][2]
            children[pidx].append(idx)
        stack.append((L, R, idx))
    # precompute factorials
    maxn = N + 5
    fact = [1] * (maxn)
    invfact = [1] * (maxn)
    for i in range(1, maxn):
        fact[i] = fact[i-1] * i % MOD
    invfact[maxn-1] = pow(fact[maxn-1], MOD-2, MOD)
    for i in range(maxn-1, 0, -1):
        invfact[i-1] = invfact[i] * i % MOD

    # we need L,R for each idx
    Ls = [0] * (N+1)
    Rs = [0] * (N+1)
    for i in range(1, N+1):
        Ls[i] = A[i-1] + 1
        Rs[i] = i
    # root 0
    Ls[0] = 1
    Rs[0] = N

    # dfs returns (size, dp)
    def dfs(u):
        segL = Ls[u]
        segR = Rs[u]
        # gather children intervals and sort by L
        ch = children[u]
        # sort children by their L
        ch.sort(key=lambda x: Ls[x])
        total_size = 0
        res = 1
        # sum sizes of real children
        for v in ch:
            sv, dv = dfs(v)
            total_size += sv
            res = res * dv % MOD
        # compute gaps
        # for u!=0, the minimal position u at R is reserved, so available = segR - segL
        # for root u==0, available = segR - segL + 1 = N
        if u == 0:
            avail = segR - segL + 1
        else:
            avail = segR - segL
        # sum of child sizes must be <= avail
        # total gaps size = avail - total_size
        # we treat each child as a part of size sv, and each gap chunk of size g but dp=1
        # number of parts = len(ch) + (#gaps with positive size), but for multinomial only sizes matter
        # multinomial: avail! / (prod sv! * prod gap_sizes! )
        res = res * fact[avail] % MOD
        # divide by child sizes factorial
        for v in ch:
            sv = (Rs[v] - Ls[v] + 1)
            # but sv is full seg size, for u!=0 that's correct; children full segs
            if u != 0:
                # for children under non-root, their segment size includes their minimal, but avail excludes u's minimal
                # still we division by sv! is correct because child segment uses sv labels
                pass
            res = res * invfact[sv] % MOD
        # now divide by gap factorials
        # compute actual gaps
        prev = segL
        parts = []
        for v in ch:
            l = Ls[v]
            # gap is [prev, l-1]
            g = l - prev
            if u != 0 and prev <= segR-1 and prev<=l-1:
                # under non-root, max position is segR-1
                g = min(g, segR - prev)
            if g > 0:
                parts.append(g)
            prev = Rs[v] + 1
        # last gap
        end = segR
        if u != 0:
            end = segR - 1
        if prev <= end:
            parts.append(end - prev + 1)
        for g in parts:
            res = res * invfact[g] % MOD
        # size of this segment = avail+1 if u!=0 else avail
        if u == 0:
            size_u = avail
        else:
            size_u = avail + 1
        return size_u, res

    ans = dfs(0)[1]
    print(ans)

if __name__ == "__main__":
    main()