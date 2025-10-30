# Read input
S = input().strip()

def min_button_presses(s):
    # Dynamic programming approach
    n = len(s)
    # dp[i] represents minimum presses needed to form string up to index i
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Empty string needs 0 presses
    
    # For each position
    for i in range(n):
        # Single digit press
        if i == 0 and s[i] == '0':
            continue  # Skip leading zeros as per constraints
        dp[i + 1] = min(dp[i + 1], dp[i] + 1)
        
        # Try pressing "00" button
        if i + 1 < n and s[i] == '0' and s[i + 1] == '0':
            dp[i + 2] = min(dp[i + 2], dp[i] + 1)
    
    return dp[n]

# Print result
print(min_button_presses(S))