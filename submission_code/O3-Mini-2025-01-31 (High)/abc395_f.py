def main():
    import sys
    input = sys.stdin.readline

    # Read input
    N, X = map(int, input().split())
    U = [0] * N
    D = [0] * N
    total = 0
    # The final sum H must be at most U[i]+D[i] for every tooth i.
    h_max_possible = None
    for i in range(N):
        u, d = map(int, input().split())
        U[i] = u
        D[i] = d
        s = u + d
        total += s
        if h_max_possible is None or s < h_max_possible:
            h_max_possible = s

    # For a candidate H, we want to decide whether we can choose (after grinding)
    # values U'_1,...,U'_N satisfying:
    #   (1) U'_i in [max(H - D[i], 0), U[i]]  (so that D'_i = H - U'_i <= D[i])
    #   (2) |U'_i - U'_{i+1}| <= X  for every i.
    #
    # We can check feasibility by “propagating” an interval.
    # For i=0, the possible U'_0 is the entire interval I_0 = [max(H-D[0], 0), U[0]].
    # For each subsequent i, the Lipschitz condition |a - b| <= X means if the previous tooth 
    # could be any value in [L, R], then the current tooth must lie in the union of intervals
    # [L - X, R + X]. Intersecting this with the tooth i’s own allowed range gives the feasible range.
    def feasible(H):
        # Start with tooth 0.
        lo = max(H - D[0], 0)
        hi = U[0]
        if lo > hi:
            return False
        # Propagate for teeth 1 to N-1.
        for i in range(1, N):
            # Tooth i’s own allowed range: U'_i must be at least H - D[i] (or 0 if negative) 
            # and at most U[i].
            cur_low = max(H - D[i], 0)
            cur_high = U[i]
            # In addition, since |U'_i - U'_{i-1}| <= X and U'_{i-1} is in [lo, hi],
            # the allowed U'_i must also lie in [lo - X, hi + X].
            new_lo = max(cur_low, lo - X)
            new_hi = min(cur_high, hi + X)
            if new_lo > new_hi:
                return False
            lo, hi = new_lo, new_hi
        return True

    # We want to minimize cost = (sum of S_i) - N*H.
    # Since the cost reduces when H is higher, we wish to choose the maximum feasible H.
    # We binary search for H in [0, h_max_possible].
    loB = 0
    hiB = h_max_possible
    best = 0
    while loB <= hiB:
        mid = (loB + hiB) // 2
        if feasible(mid):
            best = mid  # candidate H is feasible
            loB = mid + 1  # try for a larger H, which gives a lower cost
        else:
            hiB = mid - 1

    # The cost is cost = (total sum of all teeth) - N * H.
    ans = total - N * best
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()