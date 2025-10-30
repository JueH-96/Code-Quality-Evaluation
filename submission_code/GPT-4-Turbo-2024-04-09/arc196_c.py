def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    MOD = 998244353
    
    # Dynamic programming table
    # dp[i][j] represents the number of valid pairings for the first i vertices
    # where j white vertices are paired with black vertices from the first i vertices
    dp = [[0] * (N + 1) for _ in range(2 * N + 1)]
    dp[0][0] = 1
    
    # Count of white and black vertices encountered so far
    white_count = 0
    black_count = 0
    
    for i in range(2 * N):
        if S[i] == 'W':
            white_count += 1
        else:
            black_count += 1
        
        for j in range(min(white_count, black_count) + 1):
            if S[i] == 'W':
                # If current is white, it can pair with a previous black
                if j > 0:
                    dp[i + 1][j] = (dp[i + 1][j] + dp[i][j - 1]) % MOD
                # Continue without pairing
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
            else:
                # If current is black, it can be paired with a previous white
                if j < white_count:
                    dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
                # Continue without pairing
                dp[i + 1][j] = (dp[i + 1][j] + dp[i][j]) % MOD
    
    # The result is the number of ways to pair all N whites with N blacks
    print(dp[2 * N][N])