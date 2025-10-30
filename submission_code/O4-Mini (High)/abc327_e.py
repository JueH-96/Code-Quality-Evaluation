def main():
    import sys, math
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    # Read performances as floats
    P_list = [float(next(it)) for _ in range(N)]
    # A very small number to represent -infinity in our DP
    NEG = -1e100
    # decay factor
    c = 0.9
    # dp_prev[k-1][i] will be stored in dp_prev_loc[i]
    dp_prev_loc = P_list[:]      # for k=1, dp_prev[i] = P_i
    dp_cur_loc = [NEG] * N       # scratch space for next dp row
    # Precompute 1/sqrt(k) for penalties
    inv_sqrt_loc = [0.0] * (N + 1)
    for i in range(1, N + 1):
        inv_sqrt_loc[i] = 1.0 / math.sqrt(i)
    # For k=1
    A1 = max(dp_prev_loc)        # best numerator for k=1
    best = A1 - 1200.0           # b=1 for k=1, so rating = A1 - 1200/sqrt(1)
    b = 1.0                      # denominator sum of weights for k=1
    P_loc = P_list               # local alias
    # DP for k = 2..N
    for k in range(2, N + 1):
        # update denominator b_k = 1 + 0.9 * b_{k-1}
        b = b * c + 1.0
        t = k - 1
        max_prev = dp_prev_loc[0]
        A_k = NEG
        # build dp_cur_loc[i] = max weighted‐sum of length k ending at i
        for i in range(N):
            if i >= t:
                # we can extend a subsequence of length k-1 ending before i
                v = P_loc[i] + c * max_prev
                dp_cur_loc[i] = v
                if v > A_k:
                    A_k = v
            else:
                dp_cur_loc[i] = NEG
            # update prefix‐maximum of dp_prev_loc for next i
            pv = dp_prev_loc[i]
            if pv > max_prev:
                max_prev = pv
        # compute rating for this k and update best
        R_k = A_k / b - 1200.0 * inv_sqrt_loc[k]
        if R_k > best:
            best = R_k
        # swap dp_prev_loc and dp_cur_loc for next iteration
        dp_prev_loc, dp_cur_loc = dp_cur_loc, dp_prev_loc

    # print answer with sufficient precision
    sys.stdout.write("{:.15f}".format(best))

# call main
main()