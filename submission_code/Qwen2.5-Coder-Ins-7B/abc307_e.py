# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

MOD = 998244353

# Initialize the dp array
dp = [[0] * M for _ in range(N)]

# Base case: there are M ways to assign the first person
for i in range(M):
    dp[0][i] = 1

# Fill the dp array
for i in range(1, N):
    for j in range(M):
        dp[i][j] = sum(dp[i-1][k] for k in range(M) if k != j) % MOD

# The answer is the sum of all ways to assign the last person
answer = sum(dp[N-1][i] for i in range(M)) % MOD

# Print the answer
print(answer)