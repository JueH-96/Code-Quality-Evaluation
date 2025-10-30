import sys

def max_experience_points(N, A):
    dp = [[0, 0] for _ in range(N + 1)]
    for i in range(1, N + 1):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1])
        dp[i][1] = dp[i - 1][0] + A[i - 1] + (A[i - 1] if i % 2 == 0 else 0)
    return max(dp[N][0], dp[N][1])

N = int(input())
A = list(map(int, input().split()))
print(max_experience_points(N, A))