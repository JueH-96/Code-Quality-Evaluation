import sys
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
K = int(data[index])
index += 1
grid = []
for _ in range(H):
    s = data[index]
    grid.append(s)
    index += 1

ans = K + 1  # Initialize to a value larger than any possible cost

# Process all rows
for i in range(H):
    j = 0
    while j < W:
        if grid[i][j] != 'x':
            start_col = j
            end_pos_j = j
            while end_pos_j < W and grid[i][end_pos_j] != 'x':
                end_pos_j += 1
            end_col = end_pos_j - 1
            L = end_col - start_col + 1
            if L >= K:
                # Sliding window to find min number of '.' in any K-window
                sum_win = 0
                for c in range(start_col, start_col + K):
                    if grid[i][c] == '.':
                        sum_win += 1
                min_cost_block = sum_win
                for ws_col in range(start_col + 1, start_col + L - K + 1):
                    if grid[i][ws_col - 1] == '.':
                        sum_win -= 1
                    if grid[i][ws_col + K - 1] == '.':
                        sum_win += 1
                    min_cost_block = min(min_cost_block, sum_win)
                ans = min(ans, min_cost_block)
            j = end_pos_j
        else:
            j += 1

# Process all columns
for j_col in range(W):
    i_row = 0
    while i_row < H:
        if grid[i_row][j_col] != 'x':
            start_row = i_row
            end_pos_i = i_row
            while end_pos_i < H and grid[end_pos_i][j_col] != 'x':
                end_pos_i += 1
            end_row = end_pos_i - 1
            L = end_row - start_row + 1
            if L >= K:
                # Sliding window to find min number of '.' in any K-window
                sum_win = 0
                for r in range(start_row, start_row + K):
                    if grid[r][j_col] == '.':
                        sum_win += 1
                min_cost_block = sum_win
                for ws_row in range(start_row + 1, start_row + L - K + 1):
                    if grid[ws_row - 1][j_col] == '.':
                        sum_win -= 1
                    if grid[ws_row + K - 1][j_col] == '.':
                        sum_win += 1
                    min_cost_block = min(min_cost_block, sum_win)
                ans = min(ans, min_cost_block)
            i_row = end_pos_i
        else:
            i_row += 1

# Check if it was possible to find a valid window
if ans > K:
    print(-1)
else:
    print(ans)