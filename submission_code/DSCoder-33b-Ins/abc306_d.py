# YOUR CODE HERE

N = int(input())
courses = []
for _ in range(N):
    X, Y = map(int, input().split())
    courses.append((X, Y))

dp = [[0, 0] for _ in range(N+1)]
for i in range(N-1, -1, -1):
    X, Y = courses[i]
    if X == 0:
        dp[i][0] = max(dp[i+1][0], dp[i+1][1] + Y)
        dp[i][1] = dp[i+1][1]
    else:
        dp[i][0] = dp[i+1][0]
        dp[i][1] = max(dp[i+1][0] + Y, dp[i+1][1])

print(max(dp[0]))