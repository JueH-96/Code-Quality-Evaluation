import sys

# Read the input
N = int(sys.stdin.readline().strip())
sheets = []
for _ in range(N):
    A, B, C, D = map(int, sys.stdin.readline().strip().split())
    sheets.append((A, B, C, D))

# Initialize the grid
grid = [[0] * 101 for _ in range(101)]

# Mark the covered regions in the grid
for A, B, C, D in sheets:
    for x in range(A, B):
        for y in range(C, D):
            grid[x][y] += 1

# Calculate the total area
area = 0
for row in grid:
    for cell in row:
        if cell > 0:
            area += 1

print(area)