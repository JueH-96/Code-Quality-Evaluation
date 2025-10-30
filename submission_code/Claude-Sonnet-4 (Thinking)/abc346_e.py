import bisect

H, W, M = map(int, input().split())

# For each row, store (last_operation_index, color), or None if never painted
last_row_operation = [None] * (H + 1)  # 1-indexed

# For each column, store (last_operation_index, color), or None if never painted
last_col_operation = [None] * (W + 1)  # 1-indexed

for i in range(M):
    T, A, X = map(int, input().split())
    if T == 1:  # Row operation
        last_row_operation[A] = (i, X)
    else:  # Column operation
        last_col_operation[A] = (i, X)

# Find painted rows and columns with their operation times and colors
painted_rows = []  # (time, color)
painted_cols = []  # (time, color)

for r in range(1, H + 1):
    if last_row_operation[r] is not None:
        time, color = last_row_operation[r]
        painted_rows.append((time, color))

for c in range(1, W + 1):
    if last_col_operation[c] is not None:
        time, color = last_col_operation[c]
        painted_cols.append((time, color))

unpainted_rows_count = H - len(painted_rows)
unpainted_cols_count = W - len(painted_cols)

# Count colors
color_count = {}

# Region 1: unpainted rows AND unpainted columns
region1_cells = unpainted_rows_count * unpainted_cols_count
if region1_cells > 0:
    color_count[0] = color_count.get(0, 0) + region1_cells

# Region 2: painted rows AND unpainted columns
for time, color in painted_rows:
    region2_cells = unpainted_cols_count
    if region2_cells > 0:
        color_count[color] = color_count.get(color, 0) + region2_cells

# Region 3: unpainted rows AND painted columns
for time, color in painted_cols:
    region3_cells = unpainted_rows_count
    if region3_cells > 0:
        color_count[color] = color_count.get(color, 0) + region3_cells

# Region 4: painted rows AND painted columns
# Extract and sort column times
col_times = [time for time, color in painted_cols]
col_times.sort()

for row_time, row_color in painted_rows:
    # Count how many painted columns have time < row_time
    count = bisect.bisect_left(col_times, row_time)
    # These 'count' intersections will have row_color
    if count > 0:
        color_count[row_color] = color_count.get(row_color, 0) + count

# Extract and sort row times
row_times = [time for time, color in painted_rows]
row_times.sort()

for col_time, col_color in painted_cols:
    # Count how many painted rows have time < col_time
    count = bisect.bisect_left(row_times, col_time)
    # These 'count' intersections will have col_color
    if count > 0:
        color_count[col_color] = color_count.get(col_color, 0) + count

# Output
colors = sorted(color_count.keys())
print(len(colors))
for color in colors:
    print(color, color_count[color])