# def expected_value(N, K):
#     mod = 998244353
#     dp = [0] * (K + 1)
#     dp[0] = 1
#     for k in range(1, K + 1):
#         for i in range(k, -1, -1):
#             dp[i] = (dp[i] * (N - i) + dp[i - 1] * i) * 2 % mod
#     return (N * dp[K] + 1) % mod

# N, K = map(int, input().split())
# print(expected_value(N, K))