import sys
import collections

# Read all input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
M = int(data[index])
index += 1

# Initialize last times and colors for rows and columns
row_last_time = [0] * H
row_last_color = [0] * H
col_last_time = [0] * W
col_last_color = [0] * W

# Process each operation
op_idx = 1
for _ in range(M):
    T = int(data[index])
    A = int(data[index + 1])  # A is 1-based
    X = int(data[index + 2])
    index += 3
    if T == 1:  # Paint row
        row_num = A - 1  # Convert to 0-based
        row_last_time[row_num] = op_idx
        row_last_color[row_num] = X
    elif T == 2:  # Paint column
        col_num = A - 1  # Convert to 0-based
        col_last_time[col_num] = op_idx
        col_last_color[col_num] = X
    op_idx += 1  # Increment operation index

# Compute frequency of last times for rows and columns
row_freq = [0] * (M + 1)
for r in range(H):
    t = row_last_time[r]
    row_freq[t] += 1

col_freq = [0] * (M + 1)
for c in range(W):
    t = col_last_time[c]
    col_freq[t] += 1

# Compute cumulative number of rows and columns with last time <= t
cum_le_row = [0] * (M + 1)
cum_le_row[0] = row_freq[0]
for t in range(1, M + 1):
    cum_le_row[t] = cum_le_row[t - 1] + row_freq[t]

cum_le_col = [0] * (M + 1)
cum_le_col[0] = col_freq[0]
for t in range(1, M + 1):
    cum_le_col[t] = cum_le_col[t - 1] + col_freq[t]

# Group rows and columns by (last_time, last_color) for those with time > 0
row_group = collections.Counter()
for r in range(H):
    if row_last_time[r] > 0:
        t = row_last_time[r]
        color = row_last_color[r]
        row_group[(t, color)] += 1

col_group = collections.Counter()
for c in range(W):
    if col_last_time[c] > 0:
        t = col_last_time[c]
        color = col_last_color[c]
        col_group[(t, color)] += 1

# Number of never painted rows and columns
num_rows_never = row_freq[0]
num_cols_never = col_freq[0]
num_cells_both_zero = num_rows_never * num_cols_never

# Counter for color counts
color_count = collections.Counter()

# Add cells where color comes from rows (row-dominant)
for (t, color), num_rows_group in row_group.items():
    num_c_less = cum_le_col[t - 1]  # Number of columns with Tc < t
    num_cells = num_rows_group * num_c_less
    if num_cells > 0:
        color_count[color] += num_cells

# Add cells where color comes from columns (column-dominant)
for (t, color), num_cols_group in col_group.items():
    num_r_less = cum_le_row[t - 1]  # Number of rows with Tr < t
    num_cells = num_cols_group * num_r_less
    if num_cells > 0:
        color_count[color] += num_cells

# Add cells where both row and column never painted
if num_cells_both_zero > 0:
    color_count[0] += num_cells_both_zero

# Get all colors with positive count and sort them
colors = list(color_count.keys())
colors.sort()

# Output the result
K = len(colors)
print(K)
for color in colors:
    count = color_count[color]
    print(color, count)