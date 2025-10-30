# YOUR CODE HERE
import itertools

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def count_humidified_cells(H, W, D, grid, humidifier1, humidifier2):
    humidified = set()
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                if (manhattan_distance(i, j, humidifier1[0], humidifier1[1]) <= D or
                    manhattan_distance(i, j, humidifier2[0], humidifier2[1]) <= D):
                    humidified.add((i, j))
    return len(humidified)

# Read input
H, W, D = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Find all floor cells
floor_cells = [(i, j) for i in range(H) for j in range(W) if grid[i][j] == '.']

# Try all possible combinations of two floor cells
max_humidified = 0
for humidifier1, humidifier2 in itertools.combinations(floor_cells, 2):
    humidified = count_humidified_cells(H, W, D, grid, humidifier1, humidifier2)
    max_humidified = max(max_humidified, humidified)

# Print the result
print(max_humidified)