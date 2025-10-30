MOD = 998244353

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Initialize DP table
dp = [[[0] * 4 for _ in range(W)] for _ in range(H)]

for i in range(H):
    for j in range(W):
        cell = grid[i][j]
        is_fixed = cell != '?'
        
        if i == 0 and j == 0:
            if is_fixed:
                c = int(cell)
                dp[i][j][c] = 1
            else:
                for c in [1, 2, 3]:
                    dp[i][j][c] = 1
            continue
        
        if is_fixed:
            x = int(cell)
            for c in [1, 2, 3]:
                if c != x:
                    dp[i][j][c] = 0
                else:
                    sum_prev_top = 0
                    if i > 0:
                        sum_total_top = sum(dp[i-1][j][1:4])
                        sum_prev_top = sum_total_top - dp[i-1][j][c]
                    sum_prev_left = 0
                    if j > 0:
                        sum_total_left = sum(dp[i][j-1][1:4])
                        sum_prev_left = sum_total_left - dp[i][j-1][c]
                    
                    if i > 0 and j > 0:
                        current = (sum_prev_top * sum_prev_left) % MOD
                    elif i > 0:
                        current = sum_prev_top % MOD
                    elif j > 0:
                        current = sum_prev_left % MOD
                    else:
                        current = 0
                    dp[i][j][c] = current
        else:
            for c in [1, 2, 3]:
                sum_prev_top = 0
                if i > 0:
                    sum_total_top = sum(dp[i-1][j][1:4])
                    sum_prev_top = sum_total_top - dp[i-1][j][c]
                sum_prev_left = 0
                if j > 0:
                    sum_total_left = sum(dp[i][j-1][1:4])
                    sum_prev_left = sum_total_left - dp[i][j-1][c]
                
                if i > 0 and j > 0:
                    current = (sum_prev_top * sum_prev_left) % MOD
                elif i > 0:
                    current = sum_prev_top % MOD
                elif j > 0:
                    current = sum_prev_left % MOD
                else:
                    current = 0
                dp[i][j][c] = current

answer = sum(dp[H-1][W-1][1:4]) % MOD
print(answer)