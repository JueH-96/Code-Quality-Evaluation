import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    mod = 998244353

    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    # dp_sum[x] = sum of dp[j] for all j with prefix-sum S_j == x
    dp_sum = {0: 1}
    # PS = sum of dp[0..i-1], initially dp[0] = 1
    PS = 1
    S = 0
    dp_i = 0

    for a in A:
        S += a
        # number of ways j with S_j = S - K
        bad = dp_sum.get(S - K, 0)
        # dp[i] = total ways up to i-1 minus those that would form a forbidden segment ending at i
        dp_i = PS - bad
        dp_i %= mod
        # update prefix sum of dp
        PS = (PS + dp_i) % mod
        # add dp_i to dp_sum at key S
        dp_sum[S] = (dp_sum.get(S, 0) + dp_i) % mod

    # dp[N] is the number of valid divisions (since we must cut at N)
    print(dp_i % mod)

if __name__ == "__main__":
    main()