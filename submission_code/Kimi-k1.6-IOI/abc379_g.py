MOD = 998244353

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Initialize DP for the first row
dp_prev = [[0] * 4 for _ in range(W)]

for j in range(W):
    cell = grid[0][j]
    if j == 0:
        if cell != '?':
            c = int(cell)
            dp_prev[j] = [0] * 4
            dp_prev[j][c] = 1
        else:
            dp_prev[j] = [0, 1, 1, 1]
    else:
        prev_col = dp_prev[j-1]
        sum_prev = (prev_col[1] + prev_col[2] + prev_col[3]) % MOD
        current_col = [0] * 4
        cell_val = grid[0][j]
        allowed = [int(cell_val)] if cell_val != '?' else [1, 2, 3]
        for c in allowed:
            current_col[c] = (sum_prev - prev_col[c]) % MOD
        dp_prev[j] = current_col

# Process remaining rows
for i in range(1, H):
    dp_current = [[0] * 4 for _ in range(W)]
    for j in range(W):
        cell = grid[i][j]
        allowed = [int(cell)] if cell != '?' else [1, 2, 3]
        current_col = [0] * 4
        if j == 0:
            prev_row_col = dp_prev[j]
            sum_above = (prev_row_col[1] + prev_row_col[2] + prev_row_col[3]) % MOD
            for c in allowed:
                cnt = (sum_above - prev_row_col[c]) % MOD
                current_col[c] = cnt
        else:
            prev_row_col = dp_prev[j]
            sum_above = (prev_row_col[1] + prev_row_col[2] + prev_row_col[3]) % MOD
            left_col = dp_current[j-1]
            sum_left = (left_col[1] + left_col[2] + left_col[3]) % MOD
            for c in allowed:
                cnt_above = (sum_above - prev_row_col[c]) % MOD
                cnt_left = (sum_left - left_col[c]) % MOD
                current_col[c] = (cnt_left * cnt_above) % MOD
        dp_current[j] = current_col
    dp_prev = dp_current

# Sum the valid colors in the last column of the last row
result = (dp_prev[-1][1] + dp_prev[-1][2] + dp_prev[-1][3]) % MOD
print(result)