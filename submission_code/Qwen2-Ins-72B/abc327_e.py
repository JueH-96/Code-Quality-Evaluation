from math import sqrt

N = int(input())
P = list(map(int, input().split()))

dp = [0] * (N + 1)
dp2 = [0] * (N + 1)
dp3 = [0] * (N + 1)
for i in range(N):
    for j in range(N - 1, -1, -1):
        k = i + 1
        dp2[j + 1] = max(dp2[j + 1], dp[j] + P[i] * pow(0.9, k - j - 1))
        dp3[j + 1] = max(dp3[j + 1], dp3[j] + pow(0.9, k - j - 1))
        dp[j + 1] = max(dp[j + 1], dp2[j + 1] / dp3[j + 1] - 1200 / sqrt(k))

print(dp[N])