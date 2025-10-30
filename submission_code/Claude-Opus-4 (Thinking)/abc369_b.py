# YOUR CODE HERE
n = int(input())
a = []
s = []
for _ in range(n):
    ai, si = input().split()
    a.append(int(ai))
    s.append(si)

INF = float('inf')
# dp[i][j][k] = min fatigue after i-th key press
# k=0: i-th key pressed with left (left on a[i], right on j)
# k=1: i-th key pressed with right (right on a[i], left on j)
dp = [[[INF] * 2 for _ in range(101)] for _ in range(n)]

# Base case - first key press has 0 fatigue
if s[0] == 'L':
    for j in range(1, 101):
        dp[0][j][0] = 0
else:
    for j in range(1, 101):
        dp[0][j][1] = 0

# Fill DP table
for i in range(1, n):
    if s[i] == 'L':
        # Press with left hand
        for j in range(1, 101):
            # Previous key was also pressed with left hand
            dp[i][j][0] = min(dp[i][j][0], dp[i-1][j][0] + abs(a[i] - a[i-1]))
        for j in range(1, 101):
            # Previous key was pressed with right hand
            dp[i][a[i-1]][0] = min(dp[i][a[i-1]][0], dp[i-1][j][1] + abs(a[i] - j))
    else:
        # Press with right hand
        for j in range(1, 101):
            # Previous key was pressed with left hand
            dp[i][a[i-1]][1] = min(dp[i][a[i-1]][1], dp[i-1][j][0] + abs(a[i] - j))
        for j in range(1, 101):
            # Previous key was pressed with right hand
            dp[i][j][1] = min(dp[i][j][1], dp[i-1][j][1] + abs(a[i] - a[i-1]))

# Find minimum fatigue
ans = INF
for j in range(1, 101):
    ans = min(ans, dp[n-1][j][0], dp[n-1][j][1])

print(ans)