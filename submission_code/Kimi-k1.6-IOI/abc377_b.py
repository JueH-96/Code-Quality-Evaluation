# Read the grid
rows = [input().strip() for _ in range(8)]

# Determine which rows have at least one piece
row_has_piece = [False] * 8
for i in range(8):
    if '#' in rows[i]:
        row_has_piece[i] = True

# Determine which columns have at least one piece
col_has_piece = [False] * 8
for j in range(8):
    for i in range(8):
        if rows[i][j] == '#':
            col_has_piece[j] = True
            break

# Count the valid empty squares
count = 0
for i in range(8):
    for j in range(8):
        if rows[i][j] == '.' and not row_has_piece[i] and not col_has_piece[j]:
            count += 1

print(count)