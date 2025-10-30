# YOUR CODE HERE
n = int(input())
courses = []
for _ in range(n):
    x, y = map(int, input().split())
    courses.append((x, y))

# dp[i][j] = maximum tastiness after considering first i courses
# j = 0: healthy stomach, j = 1: upset stomach
# -inf means impossible state
INF = float('-inf')
dp = [[INF, INF] for _ in range(n + 1)]

# Initially healthy stomach with 0 tastiness
dp[0][0] = 0

for i in range(n):
    x, y = courses[i]
    
    # Skip the course - state doesn't change
    if dp[i][0] != INF:
        dp[i+1][0] = max(dp[i+1][0], dp[i][0])
    if dp[i][1] != INF:
        dp[i+1][1] = max(dp[i+1][1], dp[i][1])
    
    # Eat the course
    if x == 0:  # Antidotal course
        # From healthy stomach -> remains healthy
        if dp[i][0] != INF:
            dp[i+1][0] = max(dp[i+1][0], dp[i][0] + y)
        # From upset stomach -> becomes healthy
        if dp[i][1] != INF:
            dp[i+1][0] = max(dp[i+1][0], dp[i][1] + y)
    else:  # Poisonous course (x == 1)
        # From healthy stomach -> becomes upset
        if dp[i][0] != INF:
            dp[i+1][1] = max(dp[i+1][1], dp[i][0] + y)
        # From upset stomach -> dies (don't transition)

# Answer is the maximum of both states after all courses
# But we need to ensure we survive, so take max of valid states
result = max(dp[n][0], dp[n][1])
if result == INF:
    result = 0
else:
    result = max(result, 0)  # In case all courses have negative tastiness

print(result)