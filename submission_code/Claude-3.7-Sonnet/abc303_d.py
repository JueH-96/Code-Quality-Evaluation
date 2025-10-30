def min_time(X, Y, Z, S):
    n = len(S)
    INF = float('inf')
    dp = [[INF for _ in range(2)] for _ in range(n+1)]
    
    # Base case: before typing any characters (position 0)
    dp[0][0] = 0  # Initially, the Caps Lock light is off
    
    for i in range(1, n+1):
        c = S[i-1]  # The character we want at position i-1
        
        for caps_prev in range(2):
            if dp[i-1][caps_prev] == INF:  # If we can't reach this state, skip
                continue
            
            for toggle in range(2):  # 0 for don't toggle, 1 for toggle
                caps_curr = caps_prev
                toggle_cost = 0
                
                if toggle:
                    caps_curr = 1 - caps_prev
                    toggle_cost = Z
                
                # Compute the cost of typing the character
                if c == 'a':
                    type_cost = X if caps_curr == 0 else Y
                else:  # c == 'A'
                    type_cost = Y if caps_curr == 0 else X
                
                dp[i][caps_curr] = min(dp[i][caps_curr], dp[i-1][caps_prev] + toggle_cost + type_cost)
    
    # Return the minimum of the two final states
    return min(dp[n][0], dp[n][1])

# Read input
X, Y, Z = map(int, input().split())
S = input().strip()

# Compute and output the result
print(min_time(X, Y, Z, S))