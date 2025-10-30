import sys

# Read input
H, W, D = map(int, input().split())
grid = [list(input()) for _ in range(H)]

# Find all floor cells
floor_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']

# Function to calculate Manhattan distance
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# Function to find the maximum number of humidified floor cells
def max_humidified_cells():
    max_count = 0
    for i in range(len(floor_cells)):
        for j in range(i + 1, len(floor_cells)):
            x1, y1 = floor_cells[i]
            x2, y2 = floor_cells[j]
            humidified_cells = set()
            for x, y in floor_cells:
                if manhattan_distance(x, y, x1, y1) <= D or manhattan_distance(x, y, x2, y2) <= D:
                    humidified_cells.add((x, y))
            max_count = max(max_count, len(humidified_cells))
    return max_count

# Print the answer
print(max_humidified_cells())