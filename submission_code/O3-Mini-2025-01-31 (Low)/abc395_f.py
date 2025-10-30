def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    X = int(next(it))
    U = [0]*N
    D = [0]*N
    sum_list = [0]*N
    min_possible_sum = 10**18
    for i in range(N):
        u = int(next(it))
        d = int(next(it))
        U[i] = u
        D[i] = d
        s = u + d
        sum_list[i] = s
        if s < min_possible_sum:
            min_possible_sum = s

    # For a fixed H, we want to know if we can choose u_1, u_2, ..., u_N such that:
    #   - For each i, u_i is in [L_i, R_i] where
    #         L_i = max(0, H - D[i])
    #         R_i = min(H, U[i])
    #   - They satisfy the Lipschitz constraint: |u_i - u_(i+1)| <= X.
    # This is checked by propagating an interval.
    def feasible(H):
        # For tooth i, allowed range for the upper tooth's final length is [max(0, H-D[i]), min(H, U[i])].
        L0 = max(0, H - D[0])
        R0 = min(H, U[0])
        if L0 > R0:
            return False
        curr_lo, curr_hi = L0, R0
        for i in range(1, N):
            L_i = max(0, H - D[i])
            R_i = min(H, U[i])
            if L_i > R_i:
                return False
            # The previous tooth’s interval is expanded by X (both directions) due to the condition:
            # u_i must lie within [curr_lo - X, curr_hi + X] and also be within [L_i, R_i]
            new_lo = max(L_i, curr_lo - X)
            new_hi = min(R_i, curr_hi + X)
            if new_lo > new_hi:
                return False
            curr_lo, curr_hi = new_lo, new_hi
        return True

    # We want to maximize H such that the teeth can be adjusted feasibly.
    # Note: The cost is sum(U[i] + D[i]) - N * H; thus, maximizing H minimizes cost.
    lo_H = 0
    hi_H = min_possible_sum
    best_H = 0
    while lo_H <= hi_H:
        mid_H = (lo_H + hi_H) // 2
        if feasible(mid_H):
            best_H = mid_H
            lo_H = mid_H + 1
        else:
            hi_H = mid_H - 1

    total_initial = sum(sum_list)
    # The cost is the total reduction done. Since for each tooth we have:
    # Cost per tooth = (initial U[i] + D[i]) - H, so total cost = sum(initial sums) - N * H.
    result = total_initial - N * best_H
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()