def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    MOD = 998244353

    # If there are no forbidden-maximum constraints,
    # the answer is simply N! mod.
    if M == 0:
        fact = 1
        for i in range(1, N+1):
            fact = fact * i % MOD
        print(fact)
        return

    # maxL[k][r] = maximum L among all constraints (L,R,k')
    # with k'=k and R'<=r.  Means: if maxL[k][r] >= l
    # then k is forbidden to be the root of [l..r].
    maxL = [[0] * (N+1) for _ in range(N+1)]
    for _ in range(M):
        l = int(next(it))
        r = int(next(it))
        x = int(next(it))
        if maxL[x][r] < l:
            maxL[x][r] = l

    # Prefix‐max over R so that maxL[k][r] is nondecreasing in r
    for k in range(1, N+1):
        row = maxL[k]
        best = 0
        for r in range(1, N+1):
            if row[r] < best:
                row[r] = best
            else:
                best = row[r]

    # DP array: f[l][r] = # ways to build a valid permutation on interval [l..r].
    # We only need f up to N+1 in each dimension.
    f = [[0] * (N+2) for _ in range(N+2)]
    # Base case: empty interval has exactly 1 way
    for i in range(1, N+2):
        f[i][i-1] = 1

    # We'll do l from N down to 1, and r from l..N
    # so that subproblems f[l][k-1] and f[k+1][r] are ready.
    threshold = 1 << 63  # to avoid giant Python ints, do occasional mod
    for l in range(N, 0, -1):
        f_l = f[l]
        for r in range(l, N+1):
            tot = 0
            row_r = r  # just for clarity
            ml = maxL  # local alias
            f_loc = f  # local alias
            for k in range(l, r+1):
                # check if k is allowed as the maximum in [l..r]
                # exactly when maxL[k][r] < l
                if ml[k][row_r] < l:
                    tot += f_l[k-1] * f_loc[k+1][row_r]
                    # occasionally reduce so Python ints don't blow up
                    if tot >= threshold:
                        tot %= MOD
            f_l[r] = tot % MOD

    # The answer for the full [1..N] interval:
    print(f[1][N] % MOD)


# Call main()
if __name__ == "__main__":
    main()