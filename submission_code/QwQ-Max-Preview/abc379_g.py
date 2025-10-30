MOD = 998244353

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Initialize DP table: dp[i][j][k] represents the number of ways to fill cell (i,j) with value k+1
dp = [[[0] * 3 for _ in range(W)] for _ in range(H)]

for i in range(H):
    for j in range(W):
        current = grid[i][j]
        for k in range(3):
            # Check if current cell can be assigned value k+1
            if current != '?' and current != str(k+1):
                continue
            
            # Calculate sum_upper (sum of ways from cell above)
            sum_upper = 1
            if i > 0:
                sum_upper = 0
                for u in range(3):
                    if u != k:
                        sum_upper += dp[i-1][j][u]
                sum_upper %= MOD
            
            # Calculate sum_left (sum of ways from cell to the left)
            sum_left = 1
            if j > 0:
                sum_left = 0
                for l in range(3):
                    if l != k:
                        sum_left += dp[i][j-1][l]
                sum_left %= MOD
            
            # Update dp[i][j][k] with the product of sum_upper and sum_left modulo MOD
            dp[i][j][k] = (sum_upper * sum_left) % MOD

# Sum all possible values for the last cell
result = sum(dp[H-1][W-1]) % MOD
print(result)