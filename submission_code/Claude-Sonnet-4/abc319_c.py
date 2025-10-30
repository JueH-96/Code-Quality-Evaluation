from itertools import permutations
from fractions import Fraction

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

# For each possible permutation of seeing all 9 squares
total_permutations = 0
good_permutations = 0

# Generate all positions
positions = [(i, j) for i in range(3) for j in range(3)]

# Check each permutation
for perm in permutations(positions):
    total_permutations += 1
    is_good = True
    
    # Check each line
    for line in lines:
        # Find when each position in this line is seen
        line_order = []
        for pos in line:
            line_order.append((perm.index(pos), grid[pos[0]][pos[1]]))
        
        # Sort by when they're seen
        line_order.sort()
        
        # Check if this causes disappointment
        # Disappointment occurs when first two have same number, third is different
        if (line_order[0][1] == line_order[1][1] and 
            line_order[1][1] != line_order[2][1]):
            is_good = False
            break
    
    if is_good:
        good_permutations += 1

probability = good_permutations / total_permutations
print(f"{probability:.30f}")