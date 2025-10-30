n = int(input())
grid = []
for _ in range(n):
    line = input().strip()
    grid.append([int(c) for c in line])

elements = []
# Collect elements in order
# Top row (0th row in code, 1st in problem)
for j in range(n):
    elements.append(grid[0][j])
# Right column (from row 1 to n-1 in code, which is 2nd to nth in problem)
for i in range(1, n):
    elements.append(grid[i][n-1])
# Bottom row (reverse from n-2 to 0)
for j in range(n-2, -1, -1):
    elements.append(grid[n-1][j])
# Left column (from n-2 down to 1)
for i in range(n-2, 0, -1):
    elements.append(grid[i][0])

# Shift elements
if len(elements) > 0:
    new_elements = [elements[-1]] + elements[:-1]
else:
    new_elements = []

# Assign back
index = 0
# Top row
for j in range(n):
    grid[0][j] = new_elements[index]
    index += 1
# Right column
for i in range(1, n):
    grid[i][n-1] = new_elements[index]
    index += 1
# Bottom row
for j in range(n-2, -1, -1):
    grid[n-1][j] = new_elements[index]
    index += 1
# Left column
for i in range(n-2, 0, -1):
    grid[i][0] = new_elements[index]
    index += 1

# Output
for row in grid:
    print(''.join(map(str, row)))