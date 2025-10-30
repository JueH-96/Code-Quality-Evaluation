N = int(input())
courses = []
for _ in range(N):
    x, y = map(int, input().split())
    courses.append((x, y))

# dp[i][j] represents max tastiness at position i with stomach state j
# j=0: healthy, j=1: upset
dp = [[-float('inf')] * 2 for _ in range(N+1)]
dp[0][0] = 0

for i in range(N):
    x, y = courses[i]
    
    # Skip current course
    dp[i+1][0] = max(dp[i+1][0], dp[i][0])
    dp[i+1][1] = max(dp[i+1][1], dp[i][1])
    
    # Eat current course
    if x == 0:  # antidotal
        dp[i+1][0] = max(dp[i+1][0], dp[i][0] + y)  # healthy -> healthy
        dp[i+1][0] = max(dp[i+1][0], dp[i][1] + y)  # upset -> healthy
    else:  # poisonous
        dp[i+1][1] = max(dp[i+1][1], dp[i][0] + y)  # healthy -> upset
        # upset -> dead (no need to handle)

ans = max(dp[N][0], dp[N][1])
if ans == -float('inf'):
    ans = 0
print(ans)