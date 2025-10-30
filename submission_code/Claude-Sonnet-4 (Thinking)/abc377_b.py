# Read the grid
grid = []
for _ in range(8):
    grid.append(input().strip())

# Find attacked rows and columns
attacked_rows = set()
attacked_cols = set()

for i in range(8):
    for j in range(8):
        if grid[i][j] == '#':
            attacked_rows.add(i)
            attacked_cols.add(j)

# Count safe empty squares
count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and i not in attacked_rows and j not in attacked_cols:
            count += 1

print(count)