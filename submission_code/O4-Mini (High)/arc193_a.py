def main():
    import sys
    import bisect
    input = sys.stdin.readline

    N = int(input())
    W = list(map(int, input().split()))
    L = [0]*N
    R = [0]*N
    for i in range(N):
        li, ri = map(int, input().split())
        L[i] = li
        R[i] = ri

    # Build sorted-by-R list and prefix-min of W
    R_pairs = sorted((R[i], W[i]) for i in range(N))
    R_vals = [p[0] for p in R_pairs]
    prefix_minW = [10**30] * (N+1)
    for i in range(N):
        prefix_minW[i+1] = min(prefix_minW[i], R_pairs[i][1])

    # Build sorted-by-L list and suffix-min of W
    L_pairs = sorted((L[i], W[i]) for i in range(N))
    L_vals = [p[0] for p in L_pairs]
    suffix_minW = [10**30] * (N+1)
    for i in range(N-1, -1, -1):
        suffix_minW[i] = min(suffix_minW[i+1], L_pairs[i][1])

    Q = int(input())
    out = []
    INF = 10**30
    for _ in range(Q):
        s, t = map(int, input().split())
        s -= 1
        t -= 1
        ls, rs = L[s], R[s]
        lt, rt = L[t], R[t]

        # Case 1: direct edge
        if rs < lt or rt < ls:
            out.append(str(W[s] + W[t]))
            continue

        # Otherwise try a single intermediate k disjoint to both
        mn = INF

        # any interval with R_i < min(L_s, L_t)
        mnL = min(ls, lt)
        idx = bisect.bisect_left(R_vals, mnL)
        mn = min(mn, prefix_minW[idx])

        # any interval with L_i > max(R_s, R_t)
        mx = max(rs, rt)
        idx2 = bisect.bisect_right(L_vals, mx)
        mn = min(mn, suffix_minW[idx2])

        if mn >= INF//2:
            out.append("-1")
        else:
            out.append(str(W[s] + W[t] + mn))

    print("
".join(out))


if __name__ == "__main__":
    main()