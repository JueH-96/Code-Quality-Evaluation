# N = int(input())
# A = list(map(int, input().split()))
# S = [0] + list(map(int, input().split()))
# T = list(map(int, input().split()))

N = 4
A = [5, 7, 0, 3]
S = [2, 4, 5]
T = [2, 3, 2]

dp = [0] * N
dp[0] = A[0]
for i in range(1, N):
    dp[i] = max(dp[i-1] - S[i-1] + T[i-1], A[i])
print(dp[-1])