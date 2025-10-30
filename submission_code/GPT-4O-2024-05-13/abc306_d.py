# YOUR CODE HERE
def max_tastiness(N, courses):
    # dp[i][0] represents the maximum tastiness with a healthy stomach after i courses
    # dp[i][1] represents the maximum tastiness with an upset stomach after i courses
    dp = [[0, float('-inf')] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        X, Y = courses[i - 1]
        if X == 0:  # Antidotal course
            dp[i][0] = max(dp[i - 1][0] + Y, dp[i - 1][0])  # Eat or skip with healthy stomach
            dp[i][1] = max(dp[i - 1][1] + Y, dp[i - 1][1])  # Eat or skip with upset stomach
        else:  # Poisonous course
            dp[i][0] = dp[i - 1][0]  # Skip with healthy stomach
            dp[i][1] = max(dp[i - 1][0] + Y, dp[i - 1][1])  # Eat with healthy stomach or skip with upset stomach
    
    return max(dp[N][0], dp[N][1])

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
courses = [(int(data[2 * i + 1]), int(data[2 * i + 2])) for i in range(N)]

print(max_tastiness(N, courses))