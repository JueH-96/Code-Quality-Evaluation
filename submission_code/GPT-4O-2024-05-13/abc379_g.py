# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def solve():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    
    # Initialize dp array
    dp = [[[0] * 3 for _ in range(W)] for _ in range(H)]
    
    # Initialize the first cell
    if grid[0][0] == '?':
        for k in range(3):
            dp[0][0][k] = 1
    else:
        dp[0][0][int(grid[0][0]) - 1] = 1
    
    # Fill the dp array
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                continue
            if grid[i][j] == '?':
                for k in range(3):
                    if i > 0:
                        for l in range(3):
                            if k != l:
                                dp[i][j][k] = (dp[i][j][k] + dp[i-1][j][l]) % MOD
                    if j > 0:
                        for l in range(3):
                            if k != l:
                                dp[i][j][k] = (dp[i][j][k] + dp[i][j-1][l]) % MOD
            else:
                k = int(grid[i][j]) - 1
                if i > 0:
                    for l in range(3):
                        if k != l:
                            dp[i][j][k] = (dp[i][j][k] + dp[i-1][j][l]) % MOD
                if j > 0:
                    for l in range(3):
                        if k != l:
                            dp[i][j][k] = (dp[i][j][k] + dp[i][j-1][l]) % MOD
    
    # Sum up the results in the last cell
    result = sum(dp[H-1][W-1]) % MOD
    print(result)

solve()