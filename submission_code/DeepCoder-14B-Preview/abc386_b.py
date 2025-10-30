S = input().strip()
n = len(S)

if n == 0:
    print(0)
    exit()

# Initialize DP array with infinity
dp = [float('inf')] * (n + 1)
dp[0] = 0  # Base case: 0 presses needed for empty string

for i in range(1, n + 1):
    # Option 1: Press a single character button (could be any digit)
    if i >= 1:
        dp[i] = min(dp[i], dp[i-1] + 1)
    
    # Option 2: Press the '00' button if the last two characters are '00'
    if i >= 2:
        if S[i-2] == '0' and S[i-1] == '0':
            dp[i] = min(dp[i], dp[i-2] + 1)

print(dp[n])