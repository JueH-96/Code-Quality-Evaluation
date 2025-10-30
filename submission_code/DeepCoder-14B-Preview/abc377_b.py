# Read the 8x8 grid
rows = [input().strip() for _ in range(8)]

count = 0

for i in range(8):
    for j in range(8):
        if rows[i][j] != '.':
            continue
        
        # Check row i for any '#' except current cell
        row_safe = True
        for k in range(8):
            if k == j:
                continue
            if rows[i][k] == '#':
                row_safe = False
                break
        if not row_safe:
            continue
        
        # Check column j for any '#' except current cell
        col_safe = True
        for r in range(8):
            if r == i:
                continue
            if rows[r][j] == '#':
                col_safe = False
                break
        if col_safe:
            count += 1

print(count)