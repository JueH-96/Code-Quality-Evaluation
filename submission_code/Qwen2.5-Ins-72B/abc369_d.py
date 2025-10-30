# YOUR CODE HERE
import sys

def max_experience(N, A):
    dp = [0] * (N + 1)
    for i in range(1, N + 1):
        dp[i] = max(dp[i - 1], dp[i - 1] + A[i - 1] * (1 + (i % 2 == 0)))
    return dp[N]

input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(max_experience(N, A))