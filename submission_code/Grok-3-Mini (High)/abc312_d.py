import sys
# Read input from stdin
data = sys.stdin.read().strip()
S = data
n = len(S)
MOD = 998244353

# Initialize dp_prev with dp[0][bal]
dp_prev = [0] * (n + 1)
dp_prev[0] = 1

# Iterate through each character in the string
for i in range(1, n + 1):
    dp_curr = [0] * (n + 1)
    char = S[i - 1]
    # For each possible balance, compute dp_curr[bal]
    for bal in range(n + 1):
        if char == '(':
            if bal > 0:
                dp_curr[bal] = dp_prev[bal - 1]
        elif char == ')':
            if 0 <= bal + 1 <= n:
                dp_curr[bal] = dp_prev[bal + 1]
        elif char == '?':
            if bal > 0:
                dp_curr[bal] += dp_prev[bal - 1]
            if 0 <= bal + 1 <= n:
                dp_curr[bal] += dp_prev[bal + 1]
        # Take modulo for all cases
        dp_curr[bal] %= MOD
    # Copy dp_curr to dp_prev for the next iteration
    dp_prev = dp_curr[:]

# The answer is dp_prev[0], which is the number of ways with balance 0
print(dp_prev[0])