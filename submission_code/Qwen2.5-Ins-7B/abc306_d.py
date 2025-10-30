# YOUR CODE HERE
n = int(input())
courses = []
for i in range(n):
    x, y = map(int, input().split())
    courses.append((x, y))

dp = [0] * (n + 1)
dp[0] = 0
dp[1] = courses[0][1] if courses[0][0] == 0 else 0

for i in range(1, n):
    x, y = courses[i]
    if x == 0:
        dp[i + 1] = max(dp[i], dp[i - 1] + y)
    else:
        dp[i + 1] = max(dp[i], dp[i - 1])

print(dp[n])