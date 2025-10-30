import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
rectangles = [tuple(map(int, data[i:i+4])) for i in range(1, 4*N+1, 4)]

# Create a grid to keep track of covered areas
grid = [[0] * 101 for _ in range(101)]

# Mark the covered areas in the grid
for A, B, C, D in rectangles:
    for x in range(A, B):
        for y in range(C, D):
            grid[x][y] = 1

# Calculate the total covered area
covered_area = sum(sum(row) for row in grid)

print(covered_area)