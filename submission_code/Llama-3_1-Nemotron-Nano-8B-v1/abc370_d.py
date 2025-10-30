import bisect

h, w, q = map(int, input().split())

rows = [[] for _ in range(h + 1)]  # rows[1..h]
columns = [[] for _ in range(w + 1)]  # columns[1..w]

for r in range(1, h + 1):
    rows[r] = list(range(1, w + 1))
for c in range(1, w + 1):
    columns[c] = list(range(1, h + 1))

total = h * w

for _ in range(q):
    r, c = map(int, input().split())
    # Check if current cell has a wall
    row = rows[r]
    idx = bisect.bisect_left(row, c)
    present = idx < len(row) and row[idx] == c
    if present:
        del row[idx]
        col = columns[c]
        idx_r = bisect.bisect_left(col, r)
        if idx_r < len(col) and col[idx_r] == r:
            del col[idx_r]
        total -= 1
    else:
        # Collect walls from all four directions
        up_wall = None
        down_wall = None
        left_wall = None
        right_wall = None
        
        # Check up (same column, rows < r)
        col = columns[c]
        idx = bisect.bisect_left(col, r) - 1
        if idx >= 0:
            up_wall = col[idx]
        
        # Check down (same column, rows > r)
        idx = bisect.bisect_left(col, r)
        if idx < len(col):
            down_wall = col[idx]
        
        # Check left (same row, columns < c)
        row = rows[r]
        idx = bisect.bisect_left(row, c) - 1
        if idx >= 0:
            left_wall = row[idx]
        
        # Check right (same row, columns > c)
        idx = bisect.bisect_left(row, c)
        if idx < len(row):
            right_wall = row[idx]
        
        walls = set()
        if up_wall is not None:
            walls.add((up_wall, c))
        if down_wall is not None:
            walls.add((down_wall, c))
        if left_wall is not None:
            walls.add((r, left_wall))
        if right_wall is not None:
            walls.add((r, right_wall))
        
        # Destroy each wall
        for i, j in walls:
            # Remove j from row i
            row_i = rows[i]
            idx_j = bisect.bisect_left(row_i, j)
            if idx_j < len(row_i) and row_i[idx_j] == j:
                del row_i[idx_j]
                # Remove i from column j
                col_j = columns[j]
                idx_i = bisect.bisect_left(col_j, i)
                if idx_i < len(col_j) and col_j[idx_i] == i:
                    del col_j[idx_i]
                total -= 1

print(total)