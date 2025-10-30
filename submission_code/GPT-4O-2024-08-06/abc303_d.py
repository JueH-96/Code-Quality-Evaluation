# YOUR CODE HERE
def min_typing_time(X, Y, Z, S):
    n = len(S)
    # dp[i][0] -> minimum time to type first i characters with Caps Lock off
    # dp[i][1] -> minimum time to type first i characters with Caps Lock on
    dp = [[float('inf')] * 2 for _ in range(n + 1)]
    
    # Initial state, no characters typed yet
    dp[0][0] = 0  # Caps Lock off
    dp[0][1] = 0  # Caps Lock on
    
    for i in range(n):
        char = S[i]
        
        # If Caps Lock is off
        if char == 'a':
            # Type 'a' with 'a' key
            dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + X)
            # Type 'a' with Shift+'a' key (Caps Lock on)
            dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + Y)
        elif char == 'A':
            # Type 'A' with Shift+'a' key
            dp[i + 1][0] = min(dp[i + 1][0], dp[i][0] + Y)
            # Type 'A' with 'a' key (Caps Lock on)
            dp[i + 1][1] = min(dp[i + 1][1], dp[i][1] + X)
        
        # Toggle Caps Lock
        dp[i + 1][0] = min(dp[i + 1][0], dp[i + 1][1] + Z)
        dp[i + 1][1] = min(dp[i + 1][1], dp[i + 1][0] + Z)
    
    # The answer is the minimum time to type the whole string with either state
    return min(dp[n][0], dp[n][1])

import sys
input = sys.stdin.read
data = input().split()
X = int(data[0])
Y = int(data[1])
Z = int(data[2])
S = data[3]

print(min_typing_time(X, Y, Z, S))