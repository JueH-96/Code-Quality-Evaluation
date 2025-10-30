# YOUR CODE HERE
X, Y, Z = map(int, input().split())
S = input().strip()

n = len(S)
INF = float('inf')

# dp[i][j] = minimum time to type first i characters, with Caps Lock in state j
# j = 0: Caps Lock off, j = 1: Caps Lock on
dp = [[INF] * 2 for _ in range(n + 1)]
dp[0][0] = 0  # Initially, Caps Lock is off

for i in range(n):
    if S[i] == 'a':
        # Need lowercase 'a'
        if dp[i][0] < INF:
            # From Caps off state
            dp[i+1][0] = min(dp[i+1][0], dp[i][0] + X)  # Press 'a' only
            dp[i+1][1] = min(dp[i+1][1], dp[i][0] + Z + Y)  # Toggle to on, then 'a'+Shift
        if dp[i][1] < INF:
            # From Caps on state
            dp[i+1][1] = min(dp[i+1][1], dp[i][1] + Y)  # Press 'a'+Shift
            dp[i+1][0] = min(dp[i+1][0], dp[i][1] + Z + X)  # Toggle to off, then 'a' only
    else:  # S[i] == 'A'
        # Need uppercase 'A'
        if dp[i][0] < INF:
            # From Caps off state
            dp[i+1][0] = min(dp[i+1][0], dp[i][0] + Y)  # Press 'a'+Shift
            dp[i+1][1] = min(dp[i+1][1], dp[i][0] + Z + X)  # Toggle to on, then 'a' only
        if dp[i][1] < INF:
            # From Caps on state
            dp[i+1][1] = min(dp[i+1][1], dp[i][1] + X)  # Press 'a' only
            dp[i+1][0] = min(dp[i+1][0], dp[i][1] + Z + Y)  # Toggle to off, then 'a'+Shift

print(min(dp[n][0], dp[n][1]))