def solve():
    X, Y, Z = map(int, input().split())
    S = input()
    N = len(S)
    
    # dp[i][j] represents minimum time to type first i chars with caps state j
    dp = [[float('inf')] * 2 for _ in range(N + 1)]
    dp[0][0] = 0  # base case - empty string, caps off
    
    for i in range(N):
        for caps in range(2):  # 0: caps off, 1: caps on
            if dp[i][caps] == float('inf'):
                continue
                
            curr_time = dp[i][caps]
            
            # Try all possible ways to type next character
            target = S[i]
            
            # Using just 'a'
            if caps == 0 and target == 'a':
                dp[i+1][caps] = min(dp[i+1][caps], curr_time + X)
            elif caps == 1 and target == 'A':
                dp[i+1][caps] = min(dp[i+1][caps], curr_time + X)
                
            # Using shift+a
            if caps == 0 and target == 'A':
                dp[i+1][caps] = min(dp[i+1][caps], curr_time + Y)
            elif caps == 1 and target == 'a':
                dp[i+1][caps] = min(dp[i+1][caps], curr_time + Y)
                
            # Try toggling caps lock and then typing
            new_caps = 1 - caps
            if new_caps == 0 and target == 'a':
                dp[i+1][new_caps] = min(dp[i+1][new_caps], curr_time + Z + X)
            elif new_caps == 1 and target == 'A':
                dp[i+1][new_caps] = min(dp[i+1][new_caps], curr_time + Z + X)
    
    print(min(dp[N][0], dp[N][1]))

solve()