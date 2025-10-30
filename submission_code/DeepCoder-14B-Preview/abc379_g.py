MOD = 998244353

H, W = map(int, input().split())

grid = []
for _ in range(H):
    s = input().strip()
    row = []
    for c in s:
        if c == '?':
            row.append(0)  # 0 represents '?'
        else:
            row.append(int(c))
    grid.append(row)

# Initialize DP table
dp = [[[0] * 4 for _ in range(W)] for __ in range(H)]  # Using 1-based for values 1,2,3

# Base case: top-left cell (0,0)
if grid[0][0] == 0:
    for v in range(1, 4):
        dp[0][0][v] = 1
else:
    v = grid[0][0]
    dp[0][0][v] = 1

# Fill DP table
for i in range(H):
    for j in range(W):
        if i == 0 and j == 0:
            continue  # Already initialized
        for v in range(1, 4):
            if grid[i][j] != 0 and v != grid[i][j]:
                continue  # Skip if cell has a fixed value that doesn't match
            
            sum_up = 0
            if i > 0:
                for u in range(1, 4):
                    if u != v:
                        sum_up += dp[i-1][j][u]
            
            sum_left = 0
            if j > 0:
                for l in range(1, 4):
                    if l != v:
                        sum_left += dp[i][j-1][l]
            
            if i > 0 and j > 0:
                ways = sum_up * sum_left
            elif i > 0:
                ways = sum_up
            else:
                ways = sum_left
            
            dp[i][j][v] = ways % MOD

# Sum the possibilities for the bottom-right cell
answer = sum(dp[H-1][W-1][v] for v in range(1, 4)) % MOD
print(answer)