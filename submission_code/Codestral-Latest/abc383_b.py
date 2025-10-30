import sys
from itertools import combinations

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
D = int(data[2])

grid = data[3:]

# Parse the grid
floor_cells = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            floor_cells.append((i, j))

# Function to calculate the number of humidified cells
def count_humidified(cells, D):
    humidified = set()
    for cell in cells:
        for i in range(H):
            for j in range(W):
                if abs(cell[0] - i) + abs(cell[1] - j) <= D:
                    humidified.add((i, j))
    return len(humidified)

# Find the maximum number of humidified cells
max_humidified = 0
for combo in combinations(floor_cells, 2):
    max_humidified = max(max_humidified, count_humidified(combo, D))

print(max_humidified)