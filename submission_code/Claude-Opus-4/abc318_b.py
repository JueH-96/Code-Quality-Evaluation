# YOUR CODE HERE
N = int(input())
sheets = []
for _ in range(N):
    A, B, C, D = map(int, input().split())
    sheets.append((A, B, C, D))

# Create a 2D grid to mark covered cells
grid = [[False] * 101 for _ in range(101)]

# Mark all cells covered by each sheet
for A, B, C, D in sheets:
    for x in range(A, B):
        for y in range(C, D):
            grid[x][y] = True

# Count the total covered area
total_area = 0
for x in range(101):
    for y in range(101):
        if grid[x][y]:
            total_area += 1

print(total_area)