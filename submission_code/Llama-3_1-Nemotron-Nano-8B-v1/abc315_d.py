import sys
from collections import defaultdict

H, W = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(H)]

# Initialize row data
row_freq = [defaultdict(int) for _ in range(H)]
active_cells = [set(range(W)) for _ in range(H)]
colors = [set() for _ in range(H)]
for i in range(H):
    for j in range(W):
        c = grid[i][j]
        row_freq[i][c] += 1
    for c in row_freq[i]:
        if row_freq[i][c] > 0:
            colors[i].add(c)

# Initialize column data
column_freq = [defaultdict(int) for _ in range(W)]
active_cells_col = [set(range(H)) for _ in range(W)]
colors_col = [set() for _ in range(W)]
for j in range(W):
    for i in range(H):
        c = grid[i][j]
        column_freq[j][c] += 1
    for c in column_freq[j]:
        if column_freq[j][c] > 0:
            colors_col[j].add(c)

while True:
    # Mark rows and columns
    mark_rows = []
    for i in range(H):
        if len(active_cells[i]) >= 2 and len(colors[i]) == 1:
            mark_rows.append(i)
    mark_cols = []
    for j in range(W):
        if len(active_cells_col[j]) >= 2 and len(colors_col[j]) == 1:
            mark_cols.append(j)
    if not mark_rows and not mark_cols:
        break

    # Collect all to remove
    to_remove = set()
    for i in mark_rows:
        for j in active_cells[i]:
            to_remove.add((i, j))
    for j in mark_cols:
        for i in active_cells_col[j]:
            to_remove.add((i, j))
    if not to_remove:
        break

    # Remove the cells
    for i, j in to_remove:
        c = grid[i][j]

        # Update row
        row_freq[i][c] -= 1
        if row_freq[i][c] == 0:
            if c in colors[i]:
                colors[i].remove(c)
        active_cells[i].discard(j)

        # Update column
        column_freq[j][c] -= 1
        if column_freq[j][c] == 0:
            if c in colors_col[j]:
                colors_col[j].remove(c)
        active_cells_col[j].discard(i)

# Calculate remaining cookies
remaining = sum(len(active_cells[i]) for i in range(H))
print(remaining)