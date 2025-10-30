# YOUR CODE HERE
S = input().strip()
n = len(S)
dp = [float('inf')] * (n + 1)
dp[0] = 0

for i in range(1, n + 1):
    # Single digit button
    dp[i] = dp[i-1] + 1
    
    # "00" button
    if i >= 2 and S[i-2:i] == "00":
        dp[i] = min(dp[i], dp[i-2] + 1)

print(dp[n])