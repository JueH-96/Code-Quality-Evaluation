from itertools import permutations

# Read the grid
grid = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid.append(row)

# Define all lines
lines = []
# Rows
for i in range(3):
    lines.append([(i, 0), (i, 1), (i, 2)])
# Columns
for j in range(3):
    lines.append([(0, j), (1, j), (2, j)])
# Diagonals
lines.append([(0, 0), (1, 1), (2, 2)])
lines.append([(0, 2), (1, 1), (2, 0)])

# All cells
all_cells = [(i, j) for i in range(3) for j in range(3)]

# Count valid orderings
valid_count = 0

for perm in permutations(all_cells):
    disappointed = False
    
    for line in lines:
        # Find when each cell in this line is seen
        see_order = []
        for cell in line:
            see_order.append((perm.index(cell), cell))
        
        # Sort by when they're seen
        see_order.sort()
        
        # Get the values in the order they're seen
        values = [grid[cell[1][0]][cell[1][1]] for _, cell in see_order]
        
        # Check if first two have same value but third is different
        if values[0] == values[1] and values[2] != values[0]:
            disappointed = True
            break
    
    if not disappointed:
        valid_count += 1

probability = valid_count / 362880  # 9!
print(probability)