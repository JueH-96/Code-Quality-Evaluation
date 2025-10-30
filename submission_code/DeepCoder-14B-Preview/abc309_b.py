n = int(input())
grid = []
for _ in range(n):
    line = input().strip()
    grid.append(list(line))

# Generate the perimeter list
perimeter = []

# Top row
for j in range(n):
    perimeter.append((0, j))

# Right column
for i in range(1, n-1):
    perimeter.append((i, n-1))

# Bottom row
for j in range(n-1, -1, -1):
    perimeter.append((n-1, j))

# Left column
for i in range(n-2, 0, -1):
    perimeter.append((i, 0))

# Extract the values in the order of the perimeter
extract_values = [grid[i][j] for (i, j) in perimeter]

# Shift the values
if len(extract_values) > 0:
    shifted_values = [extract_values[-1]] + extract_values[:-1]
else:
    shifted_values = extract_values.copy()  # Handle empty case if needed

# Assign the shifted values back to the grid
for k in range(len(perimeter)):
    i, j = perimeter[k]
    grid[i][j] = shifted_values[k]

# Print the resulting grid
for row in grid:
    print(''.join(row))