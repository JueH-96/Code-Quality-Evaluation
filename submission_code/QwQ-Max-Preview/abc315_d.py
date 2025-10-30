h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]
present = [[True for _ in range(w)] for _ in range(h)]

while True:
    row_marks = set()
    # Check rows
    for i in range(h):
        present_cols = [j for j in range(w) if present[i][j]]
        if len(present_cols) < 2:
            continue
        first_j = present_cols[0]
        color = grid[i][first_j]
        all_same = True
        for j in present_cols[1:]:
            if grid[i][j] != color:
                all_same = False
                break
        if all_same:
            for j in present_cols:
                row_marks.add((i, j))
    
    # Check columns
    column_marks = set()
    for j in range(w):
        present_rows = [i for i in range(h) if present[i][j]]
        if len(present_rows) < 2:
            continue
        first_i = present_rows[0]
        color = grid[first_i][j]
        all_same = True
        for i in present_rows[1:]:
            if grid[i][j] != color:
                all_same = False
                break
        if all_same:
            for i in present_rows:
                column_marks.add((i, j))
    
    total_marks = row_marks.union(column_marks)
    if not total_marks:
        break
    for (i, j) in total_marks:
        present[i][j] = False

# Count remaining cookies
remaining = sum(sum(row) for row in present)
print(remaining)