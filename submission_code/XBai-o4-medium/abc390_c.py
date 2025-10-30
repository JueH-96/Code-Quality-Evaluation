import sys

h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]

black_cells = [(i, j) for i in range(h) for j in range(w) if grid[i][j] == '#']

if not black_cells:
    print("No")
    sys.exit()

min_row = min(i for i, j in black_cells)
max_row = max(i for i, j in black_cells)
min_col = min(j for i, j in black_cells)
max_col = max(j for i, j in black_cells)

for i in range(h):
    for j in range(w):
        if min_row <= i <= max_row and min_col <= j <= max_col:
            if grid[i][j] == '.':
                print("No")
                sys.exit()
        else:
            if grid[i][j] == '#':
                print("No")
                sys.exit()

print("Yes")