import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = []
B = []
X = []
for i in range(N-1):
    A.append(int(data[3*i+1]))
    B.append(int(data[3*i+2]))
    X.append(int(data[3*i+3]))

dp = [float('inf')] * (N+1)
dp[1] = 0

for i in range(1, N):
    dp[i+1] = min(dp[i+1], dp[i] + A[i-1])
    dp[X[i-1]] = min(dp[X[i-1]], dp[i] + B[i-1])

print(dp[N])