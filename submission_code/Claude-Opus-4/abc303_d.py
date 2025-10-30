# YOUR CODE HERE
X, Y, Z = map(int, input().split())
S = input().strip()

n = len(S)
# dp[i][j] = minimum time to type first i characters with caps lock state j (0=off, 1=on)
INF = float('inf')
dp = [[INF] * 2 for _ in range(n + 1)]

# Initially, we have typed 0 characters and caps lock is off
dp[0][0] = 0

for i in range(n):
    char = S[i]
    
    # From caps lock OFF state
    if dp[i][0] < INF:
        if char == 'a':
            # Type 'a' directly (caps off + 'a' = 'a')
            dp[i+1][0] = min(dp[i+1][0], dp[i][0] + X)
            # Toggle caps and type 'a' (caps on + 'a' = 'A', wrong)
            # So we need to use shift: caps on + shift+'a' = 'a'
            dp[i+1][1] = min(dp[i+1][1], dp[i][0] + Z + Y)
        else:  # char == 'A'
            # Type 'A' with shift (caps off + shift+'a' = 'A')
            dp[i+1][0] = min(dp[i+1][0], dp[i][0] + Y)
            # Toggle caps and type 'a' (caps on + 'a' = 'A')
            dp[i+1][1] = min(dp[i+1][1], dp[i][0] + Z + X)
    
    # From caps lock ON state
    if dp[i][1] < INF:
        if char == 'a':
            # Type 'a' with shift (caps on + shift+'a' = 'a')
            dp[i+1][1] = min(dp[i+1][1], dp[i][1] + Y)
            # Toggle caps and type 'a' (caps off + 'a' = 'a')
            dp[i+1][0] = min(dp[i+1][0], dp[i][1] + Z + X)
        else:  # char == 'A'
            # Type 'A' directly (caps on + 'a' = 'A')
            dp[i+1][1] = min(dp[i+1][1], dp[i][1] + X)
            # Toggle caps and type 'A' with shift (caps off + shift+'a' = 'A')
            dp[i+1][0] = min(dp[i+1][0], dp[i][1] + Z + Y)

print(min(dp[n][0], dp[n][1]))