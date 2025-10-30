def solve():
    X, Y, Z = map(int, input().split())
    S = input()
    n = len(S)
    
    dp = [[float('inf')] * 2 for _ in range(n + 1)]
    dp[0][0] = 0
    
    for i in range(n):
        # Case 1: Press 'a' key
        if S[i] == 'a':
            dp[i+1][0] = min(dp[i+1][0], dp[i][0] + X)
            dp[i+1][1] = min(dp[i+1][1], dp[i][1] + Y)
        else:
            dp[i+1][0] = min(dp[i+1][0], dp[i][0] + Y)
            dp[i+1][1] = min(dp[i+1][1], dp[i][1] + X)
            
        # Case 2: Press Caps Lock key
        dp[i][1] = min(dp[i][1], dp[i][0] + Z)
        dp[i][0] = min(dp[i][0], dp[i][1] + Z)
        
    print(min(dp[n][0], dp[n][1]))

solve()