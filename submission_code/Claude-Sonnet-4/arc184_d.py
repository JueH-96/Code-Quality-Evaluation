n = int(input())
balls = []
for i in range(n):
    x, y = map(int, input().split())
    balls.append((x, y))

# Sort by x coordinate
balls.sort()

# Extract y coordinates
y_coords = [ball[1] for ball in balls]

MOD = 998244353

# dp[i][last_y] = number of ways to choose from first i balls where last chosen ball has y-coordinate last_y
# But this might be too memory intensive

# Alternative: for each position, we can either include it or not
# If we include position i, then all previously included positions must have y > y_coords[i]

# Let's use a different DP approach
# dp[i] = number of valid antichains ending at position i
dp = [0] * n
count = [0] * (n + 2)  # count[y] = number of antichains ending with y-coordinate y

result = 1  # empty set

for i in range(n):
    y = y_coords[i]
    
    # Count antichains that can be extended with current ball
    # We can extend any antichain ending with y' > y
    ways = 1  # just this ball alone
    for y_prev in range(y + 1, n + 1):
        ways = (ways + count[y_prev]) % MOD
    
    dp[i] = ways
    count[y] = (count[y] + ways) % MOD
    result = (result + ways) % MOD

print(result)