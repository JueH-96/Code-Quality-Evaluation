n = int(input())
a = list(map(int, input().split()))

# dp[i][j] = maximum experience points after considering first i monsters
# and having defeated exactly j of them
# We need at most n defeated monsters
dp = [[-1] * (n + 1) for _ in range(n + 1)]

# Base case: no monsters considered, 0 defeated
dp[0][0] = 0

# Fill the DP table
for i in range(n):
    for j in range(i + 2):  # Can't defeat more monsters than we've seen
        if dp[i][j] == -1:
            continue
            
        # Option 1: Let the current monster go
        if dp[i + 1][j] < dp[i][j]:
            dp[i + 1][j] = dp[i][j]
        
        # Option 2: Defeat the current monster
        if j < n:
            points = a[i]
            # If this will be an even-numbered defeated monster
            if (j + 1) % 2 == 0:
                points *= 2
            
            if dp[i + 1][j + 1] < dp[i][j] + points:
                dp[i + 1][j + 1] = dp[i][j] + points

# Find the maximum experience points
max_exp = 0
for j in range(n + 1):
    if dp[n][j] > max_exp:
        max_exp = dp[n][j]

print(max_exp)