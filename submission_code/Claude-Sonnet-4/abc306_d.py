n = int(input())
courses = []
for _ in range(n):
    x, y = map(int, input().split())
    courses.append((x, y))

# dp[i][state] = maximum tastiness at course i with given state
# state 0: healthy, state 1: upset
# Use -float('inf') to represent impossible states
dp = [[-float('inf'), -float('inf')] for _ in range(n + 1)]
dp[0][0] = 0  # Start with healthy stomach and 0 tastiness

for i in range(n):
    x, y = courses[i]
    
    # If we have a healthy stomach at course i
    if dp[i][0] != -float('inf'):
        # Skip this course - remain healthy
        dp[i + 1][0] = max(dp[i + 1][0], dp[i][0])
        
        # Eat this course
        if x == 0:  # Antidotal course
            # healthy → healthy
            dp[i + 1][0] = max(dp[i + 1][0], dp[i][0] + y)
        else:  # Poisonous course
            # healthy → upset
            dp[i + 1][1] = max(dp[i + 1][1], dp[i][0] + y)
    
    # If we have an upset stomach at course i
    if dp[i][1] != -float('inf'):
        # Skip this course - remain upset
        dp[i + 1][1] = max(dp[i + 1][1], dp[i][1])
        
        # Eat this course
        if x == 0:  # Antidotal course
            # upset → healthy
            dp[i + 1][0] = max(dp[i + 1][0], dp[i][1] + y)
        # If x == 1 (poisonous), eating would cause death, so we don't consider it

# The answer is the maximum of healthy or upset stomach at the end
# Both states are valid for survival
result = max(dp[n][0], dp[n][1])
if result == -float('inf'):
    result = 0

print(result)