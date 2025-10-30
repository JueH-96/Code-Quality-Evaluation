import sys
def main():
    import sys
    readline = sys.stdin.readline
    N, X = map(int, readline().split())
    U = [0] * N
    D = [0] * N
    tot_S = 0
    S_min = 1 << 62
    for i in range(N):
        u, d = map(int, readline().split())
        U[i] = u
        D[i] = d
        s = u + d
        tot_S += s
        if s < S_min:
            S_min = s

    # Check if it's possible to choose u[i] in [L_i(H), R_i(H)]
    # with |u[i] - u[i-1]| <= X for all i.
    def feasible(H):
        # initial interval for u[0]
        low = H - D[0]
        if low < 0:
            low = 0
        high = U[0] if U[0] < H else H
        if low > high:
            return False
        x = X
        U_loc = U
        D_loc = D
        # propagate intervals
        for i in range(1, N):
            # allowed by sum constraint
            Li = H - D_loc[i]
            if Li < 0:
                Li = 0
            Ri = U_loc[i] if U_loc[i] < H else H
            # expand previous by X (Lipschitz)
            new_low = low - x
            if new_low < Li:
                new_low = Li
            new_high = high + x
            if new_high > Ri:
                new_high = Ri
            if new_low > new_high:
                return False
            low, high = new_low, new_high
        return True

    # binary search for maximum H in [0, S_min]
    lo = 0
    hi = S_min
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1

    H_max = lo
    # total cost = sum(S_i) - N * H_max
    answer = tot_S - H_max * N
    print(answer)

if __name__ == "__main__":
    main()