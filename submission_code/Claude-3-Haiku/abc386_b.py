import sys

def min_button_presses(s):
    n = len(s)
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        for j in range(max(0, i - 10), i):
            if s[j] != '0' or i - j >= 2:
                dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n]

s = input().strip()
print(min_button_presses(s))