# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    H = int(data[0])
    W = int(data[1])
    S = data[2:]
    
    # dp[i][j][k] means the number of ways to fill the grid up to (i, j) with S[i][j] = k
    dp = [[[0] * 4 for _ in range(W)] for _ in range(H)]
    
    # Initialize the first cell
    if S[0][0] == '?':
        dp[0][0][1] = 1
        dp[0][0][2] = 1
        dp[0][0][3] = 1
    else:
        dp[0][0][int(S[0][0])] = 1
    
    # Fill the dp table
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                continue
            if S[i][j] == '?':
                possible_values = [1, 2, 3]
            else:
                possible_values = [int(S[i][j])]
            
            for k in possible_values:
                if i > 0:
                    for prev_k in range(1, 4):
                        if prev_k != k:
                            dp[i][j][k] = (dp[i][j][k] + dp[i-1][j][prev_k]) % MOD
                if j > 0:
                    for prev_k in range(1, 4):
                        if prev_k != k:
                            dp[i][j][k] = (dp[i][j][k] + dp[i][j-1][prev_k]) % MOD
    
    # Calculate the result
    result = 0
    for k in range(1, 4):
        result = (result + dp[H-1][W-1][k]) % MOD
    
    print(result)