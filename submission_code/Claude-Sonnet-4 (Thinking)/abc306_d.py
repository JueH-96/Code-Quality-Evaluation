n = int(input())
courses = []
for _ in range(n):
    x, y = map(int, input().split())
    courses.append((x, y))

# dp[i][state] = max tastiness from course i onwards with given stomach state at the start of course i
# state 0 = healthy, state 1 = upset
dp = [[0] * 2 for _ in range(n + 2)]

# Fill dp table backwards
for i in range(n, 0, -1):
    x, y = courses[i - 1]  # 0-indexed
    
    # Healthy stomach at the start of course i
    skip = dp[i + 1][0]
    if x == 0:  # antidotal, stays healthy
        eat = y + dp[i + 1][0]
    else:  # poisonous, becomes upset
        eat = y + dp[i + 1][1]
    dp[i][0] = max(skip, eat)
    
    # Upset stomach at the start of course i
    skip = dp[i + 1][1]
    if x == 0:  # antidotal, becomes healthy
        eat = y + dp[i + 1][0]
    else:  # poisonous, dies
        eat = float('-inf')
    dp[i][1] = max(skip, eat)

print(dp[1][0])