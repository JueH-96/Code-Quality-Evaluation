MOD = 998244353

H, W = map(int, input().split())
grid = []
for _ in range(H):
    s = input().strip()
    row = []
    for c in s:
        if c == '?':
            row.append(-1)
        else:
            row.append(int(c) - 1)
    grid.append(row)

# Initialize DP table
dp = [[[0 for _ in range(3)] for __ in range(W)] for ___ in range(H)]

for i in range(H):
    for j in range(W):
        for c in range(3):
            if grid[i][j] != -1 and c != grid[i][j]:
                dp[i][j][c] = 0
                continue
            
            ways_left = 1
            if j > 0:
                ways_left = 0
                for left_c in range(3):
                    if left_c != c:
                        ways_left += dp[i][j-1][left_c]
                ways_left %= MOD
            
            ways_above = 1
            if i > 0:
                ways_above = 0
                for above_c in range(3):
                    if above_c != c:
                        ways_above += dp[i-1][j][above_c]
                ways_above %= MOD
            
            dp[i][j][c] = (ways_left * ways_above) % MOD

total = sum(dp[H-1][W-1][c] for c in range(3)) % MOD
print(total)