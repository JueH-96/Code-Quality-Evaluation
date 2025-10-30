# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
courses = []
for i in range(1, 2 * N, 2):
    X = int(data[i])
    Y = int(data[i + 1])
    courses.append((X, Y))

dp = [[0, 0] for _ in range(N + 1)]
for i in range(N):
    X, Y = courses[i]
    if X == 0:
        dp[i + 1][0] = max(dp[i][0], dp[i][1])
        dp[i + 1][1] = dp[i][0] + Y
    else:
        dp[i + 1][0] = dp[i][1]
        dp[i + 1][1] = max(dp[i][0], dp[i][1]) + Y

print(max(dp[N][0], dp[N][1]))