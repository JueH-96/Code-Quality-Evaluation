# YOUR CODE HERE
from itertools import permutations

# Read the grid
grid = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid.append(row)

# Define all lines (rows, columns, diagonals)
lines = []
# Rows
for i in range(3):
    lines.append([(i, 0), (i, 1), (i, 2)])
# Columns
for j in range(3):
    lines.append([(0, j), (1, j), (2, j)])
# Diagonals
lines.append([(0, 0), (1, 1), (2, 2)])
lines.append([(2, 0), (1, 1), (0, 2)])

# All cells
cells = [(i, j) for i in range(3) for j in range(3)]

# Count valid orderings
valid_count = 0
total_count = 0

for perm in permutations(cells):
    total_count += 1
    disappointed = False
    
    # For each line, check if it causes disappointment
    for line in lines:
        # Find when each cell in this line is seen
        positions = []
        for cell in line:
            positions.append(perm.index(cell))
        
        # Sort to get the order in which cells of this line are seen
        sorted_positions = sorted(enumerate(positions), key=lambda x: x[1])
        
        # Get the cells in the order they are seen
        first_idx = sorted_positions[0][0]
        second_idx = sorted_positions[1][0]
        third_idx = sorted_positions[2][0]
        
        first_cell = line[first_idx]
        second_cell = line[second_idx]
        third_cell = line[third_idx]
        
        # Check if first two have same value but third is different
        if (grid[first_cell[0]][first_cell[1]] == grid[second_cell[0]][second_cell[1]] and
            grid[first_cell[0]][first_cell[1]] != grid[third_cell[0]][third_cell[1]]):
            disappointed = True
            break
    
    if not disappointed:
        valid_count += 1

# Calculate probability
probability = valid_count / total_count
print(f"{probability:.30f}")