import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i + 1] = prefix_sums[i] + A[i]
    dp = [0] * (N + 1)
    dp[0] = 1
    prefix_dp = [0] * (N + 1)
    prefix_dp[0] = 1
    sum_map = {}
    sum_map[0] = 1
    for i in range(1, N + 1):
        S_i = prefix_sums[i]
        subtract = sum_map.get(S_i - K, 0)
        dp[i] = (prefix_dp[i - 1] - subtract) % MOD
        prefix_dp[i] = (prefix_dp[i - 1] + dp[i]) % MOD
        sum_map[S_i] = (sum_map.get(S_i, 0) + dp[i]) % MOD
    print(dp[N] % MOD)

if __name__ == "__main__":
    main()