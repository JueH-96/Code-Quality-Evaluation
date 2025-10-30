import sys
import threading
def main():
    import sys

    input = sys.stdin.readline
    N, K = map(int, input().split())
    A = set(map(int, input().split()))
    # Build the list of remaining socks by color
    socks = []
    for color in range(1, N+1):
        cnt = 2 - (1 if color in A else 0)
        if cnt > 0:
            socks.extend([color]*cnt)
    M = len(socks)
    P = M // 2  # number of pairs we must form

    # If P == 0, no pairs, cost zero.
    if P == 0:
        print(0)
        return

    # We will do a parametric DP in which for a given penalty X per pair
    # we compute:
    #   dp[i] = minimum of (sum of (cost - X) over chosen pairs) up to i-th sock
    #   cp[i] = number of pairs chosen in that optimum
    # Then we binary‐search X until cp[M] >= P but cp[M] for X+1 < P.
    # Finally answer = dp[M] + X * P.

    def dp_with_penalty(X):
        # Rolling arrays for dp and count
        # dp[i] only depends on dp[i-1] and dp[i-2]
        dp_im2 = 0
        cp_im2 = 0
        dp_im1 = 0
        cp_im1 = 0
        # we'll build up to dp[i]
        for i in range(2, M+1):
            # option1: skip sock i-1, carry dp_im1
            best_dp = dp_im1
            best_cp = cp_im1
            # option2: pair sock i-2 and i-1
            cost = socks[i-1] - socks[i-2] - X
            cand_dp = dp_im2 + cost
            cand_cp = cp_im2 + 1
            # choose better
            if cand_dp < best_dp or (cand_dp == best_dp and cand_cp > best_cp):
                best_dp = cand_dp
                best_cp = cand_cp
            # shift
            dp_im2, cp_im2 = dp_im1, cp_im1
            dp_im1, cp_im1 = best_dp, best_cp
        # handle M==1 edge
        if M == 1:
            return 0, 0
        return dp_im1, cp_im1

    # Binary search on X
    lo, hi = 0, N  # penalty range
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, cnt = dp_with_penalty(mid)
        if cnt >= P:
            lo = mid
        else:
            hi = mid - 1

    final_dp, final_cp = dp_with_penalty(lo)
    # final_dp = sum(cost - lo) over chosen pairs = actual_cost - lo*final_cp
    # We forced final_cp >= P but for lo+1 it falls below, so final_cp == P
    result = final_dp + lo * P
    print(result)


if __name__ == "__main__":
    main()