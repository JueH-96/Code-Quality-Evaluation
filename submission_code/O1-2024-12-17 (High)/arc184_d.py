def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    MOD = 998244353

    N = int(input())
    # We will create an array p of length N where p[x-1] = y.
    # This effectively re-labels the balls in ascending order of x
    # and stores their y-values in that order.
    p = [0] * N
    for _ in range(N):
        x, y = map(int, input().split())
        # x and y are 1-based; store y at index x-1
        p[x - 1] = y

    # Build a Sparse Table to do RMQ (range-minimum-query) on p.
    # We need, for any interval [L..R], to find the index of the minimum p_i in O(1).
    import math
    LOG = int(math.log2(N)) + 1
    st = [[0] * N for _ in range(LOG)]
    # st[k][i] will store the index of the minimum p-value in the interval [i, i + 2^k - 1].

    # Initialize k = 0 (intervals of length 1)
    for i in range(N):
        st[0][i] = i

    length_power = 1
    k = 1
    while (1 << k) <= N:
        length_power = 1 << k  # interval length
        half = 1 << (k - 1)
        for i in range(N - length_power + 1):
            left_index = st[k - 1][i]
            right_index = st[k - 1][i + half]
            # Compare p[left_index], p[right_index] to decide which index
            if p[left_index] <= p[right_index]:
                st[k][i] = left_index
            else:
                st[k][i] = right_index
        k += 1

    def rmq(L, R):
        """
        Return the index of the minimum element p[i] in p[L..R].
        """
        length = R - L + 1
        k = length.bit_length() - 1
        left_idx = st[k][L]
        right_idx = st[k][R - (1 << k) + 1]
        if p[left_idx] <= p[right_idx]:
            return left_idx
        else:
            return right_idx

    # dp[L][R] will store the number of possible "final subsets"
    # in the sub-array p[L..R] (0-based), under the partial-order rules given.
    dp = [[0] * N for _ in range(N)]

    # A helper that returns dp[L][R] if L <= R, or 1 if L>R (empty sub-range => exactly 1 "subset": the empty one)
    def get_dp(L, R):
        if L > R:
            return 1
        return dp[L][R]

    # Base cases: sub-interval of length 1 => dp[L][L] = 1
    for i in range(N):
        dp[i][i] = 1

    # Fill dp in increasing order of sub-interval length
    for length in range(2, N + 1):
        for L in range(N - length + 1):
            R = L + length - 1
            i_min = rmq(L, R)  # index of the minimal p-value in p[L..R]
            if i_min == R:
                # If the minimum is at the rightmost position,
                # dp[L][R] = dp[L][R-1].
                dp[L][R] = get_dp(L, R - 1) % MOD
            else:
                # Otherwise dp[L][R] = dp[L..i_min-1] * (2 + dp[i_min+1..R])
                left_val = get_dp(L, i_min - 1) % MOD
                right_val = get_dp(i_min + 1, R) % MOD
                dp[L][R] = (left_val * ((2 + right_val) % MOD)) % MOD

    # Our final answer is dp[0][N-1], i.e. the number of final subsets for the whole range
    print(dp[0][N - 1] % MOD)

# Don't forget to call main().
if __name__ == "__main__":
    main()