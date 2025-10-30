def min_button_presses(S):
    n = len(S)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(n):
        # Single digit press
        dp[i + 1] = min(dp[i + 1], dp[i] + 1)
        
        # Double zero press if applicable
        if i < n - 1 and S[i] == '1' and S[i + 1] == '0':
            j = i + 1
            while j < n and S[j] == '0':
                j += 1
            dp[j] = min(dp[j], dp[i] + 1)
    
    return dp[n]

import sys
input = sys.stdin.read
S = input().strip()
print(min_button_presses(S))