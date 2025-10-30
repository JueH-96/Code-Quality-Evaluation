import sys

# Read H and W
H, W = map(int, sys.stdin.readline().split())

# Read the grid
grid = [sys.stdin.readline().strip() for _ in range(H)]

# Calculate row counts
row_counts = [0] * H
for i in range(H):
    row_counts[i] = grid[i].count('#')

# Calculate column counts
col_counts = [0] * W
for j in range(W):
    for i in range(H):
        if grid[i][j] == '#':
            col_counts[j] += 1

# Find the row with the minimum positive count
# Collect all positive row counts
positive_row_counts = [count for count in row_counts if count > 0]
# Find the minimum among them
min_row_count = min(positive_row_counts)
# Find the index (0-based) of the first occurrence of this minimum count
# The problem guarantees a unique answer, so this minimum count appears exactly once among positive counts.
i_snuke_0based = row_counts.index(min_row_count)

# Find the column with the minimum positive count
# Collect all positive column counts
positive_col_counts = [count for count in col_counts if count > 0]
# Find the minimum among them
min_col_count = min(positive_col_counts)
# Find the index (0-based) of the first occurrence of this minimum count
# The problem guarantees a unique answer, so this minimum count appears exactly once among positive counts.
j_snuke_0based = col_counts.index(min_col_count)

# Print the result (1-based indices)
print(i_snuke_0based + 1, j_snuke_0based + 1)