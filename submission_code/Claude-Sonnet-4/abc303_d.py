X, Y, Z = map(int, input().split())
S = input().strip()

n = len(S)

# dp[i][caps] = minimum cost to type first i characters with caps state
# caps = 0 means caps off, caps = 1 means caps on
dp = [[float('inf')] * 2 for _ in range(n + 1)]

# Initially caps is off
dp[0][0] = 0

for i in range(n):
    char = S[i]
    
    for caps in range(2):
        if dp[i][caps] == float('inf'):
            continue
            
        # Option 1: Use 'a' key only
        if (caps == 0 and char == 'a') or (caps == 1 and char == 'A'):
            # Direct match with current caps state
            dp[i + 1][caps] = min(dp[i + 1][caps], dp[i][caps] + X)
        
        # Option 2: Use 'a' + Shift
        if (caps == 0 and char == 'A') or (caps == 1 and char == 'a'):
            # Shift gives opposite case
            dp[i + 1][caps] = min(dp[i + 1][caps], dp[i][caps] + Y)
        
        # Option 3: Toggle caps lock first, then type
        new_caps = 1 - caps
        
        # After toggling, use 'a' key only
        if (new_caps == 0 and char == 'a') or (new_caps == 1 and char == 'A'):
            dp[i + 1][new_caps] = min(dp[i + 1][new_caps], dp[i][caps] + Z + X)
        
        # After toggling, use 'a' + Shift
        if (new_caps == 0 and char == 'A') or (new_caps == 1 and char == 'a'):
            dp[i + 1][new_caps] = min(dp[i + 1][new_caps], dp[i][caps] + Z + Y)

print(min(dp[n][0], dp[n][1]))