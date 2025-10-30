# Read the 9x9 grid from input
grid = []
for _ in range(9):
    row = list(map(int, input().split()))
    grid.append(row)

# Define the set containing numbers 1 through 9
full_set = set(range(1, 10))

# Check rows
for row in grid:
    if set(row) != full_set:
        print("No")
        exit()

# Check columns
for col in zip(*grid):
    if set(col) != full_set:
        print("No")
        exit()

# Check 3x3 subgrids
for i in range(0, 9, 3):
    for j in range(0, 9, 3):
        # Collect numbers in the 3x3 subgrid
        subgrid = []
        for row in grid[i:i+3]:
            subgrid.extend(row[j:j+3])
        if set(subgrid) != full_set:
            print("No")
            exit()

# If all checks pass
print("Yes")