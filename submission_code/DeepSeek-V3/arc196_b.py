h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]

min_row = h
max_row = -1
min_col = w
max_col = -1

# Find the boundaries of existing black cells
for i in range(h):
    for j in range(w):
        if grid[i][j] == '#':
            if i < min_row:
                min_row = i
            if i > max_row:
                max_row = i
            if j < min_col:
                min_col = j
            if j > max_col:
                max_col = j

# Check if all existing black cells are within the rectangle and no existing white cells inside
possible = True
for i in range(h):
    for j in range(w):
        cell = grid[i][j]
        if (min_row <= i <= max_row) and (min_col <= j <= max_col):
            # Inside the rectangle, must be black or ?
            if cell == '.':
                possible = False
                break
        else:
            # Outside the rectangle, must be white or ?
            if cell == '#':
                possible = False
                break
    if not possible:
        break

if possible:
    print("Yes")
else:
    print("No")