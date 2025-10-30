import sys
import collections

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
W = int(data[index + 1])
index += 2
grid_rows = data[index:index + H]
grid = [list(row) for row in grid_rows]

# Initialize counters and totals
row_counter = [collections.Counter() for _ in range(H)]
col_counter = [collections.Counter() for _ in range(W)]
row_total = [0] * H
col_total = [0] * W

for i in range(H):
    for j in range(W):
        c = grid[i][j]
        row_counter[i][c] += 1
        col_counter[j][c] += 1
        row_total[i] += 1
        col_total[j] += 1

# Perform the procedure
while True:
    # Find marked rows and columns
    marked_rows = []
    for i in range(H):
        if len(row_counter[i]) == 1 and row_total[i] >= 2:
            marked_rows.append(i)
    marked_cols = []
    for j in range(W):
        if len(col_counter[j]) == 1 and col_total[j] >= 2:
            marked_cols.append(j)
    
    if not marked_rows and not marked_cols:
        break
    
    # Remove cells in marked rows first
    for i in marked_rows:
        for j in range(W):
            if grid[i][j] != '.':
                c = grid[i][j]
                # Update row counter and total
                row_counter[i][c] -= 1
                if row_counter[i][c] <= 0:
                    del row_counter[i][c]
                row_total[i] -= 1
                # Update column counter and total
                col_counter[j][c] -= 1
                if col_counter[j][c] <= 0:
                    del col_counter[j][c]
                col_total[j] -= 1
                # Mark as removed
                grid[i][j] = '.'
    
    # Remove cells in marked columns, only if not already removed
    for j in marked_cols:
        for i in range(H):
            if grid[i][j] != '.':
                c = grid[i][j]
                # Update row counter and total
                row_counter[i][c] -= 1
                if row_counter[i][c] <= 0:
                    del row_counter[i][c]
                row_total[i] -= 1
                # Update column counter and total
                col_counter[j][c] -= 1
                if col_counter[j][c] <= 0:
                    del col_counter[j][c]
                col_total[j] -= 1
                # Mark as removed
                grid[i][j] = '.'

# Calculate and output the number of remaining cookies
remaining = sum(row_total)
print(remaining)