import itertools

# Read the grid
grid = []
for _ in range(3):
    row = list(map(int, input().split()))
    grid.append(row)

# Define all possible lines (rows, columns, diagonals)
lines = [
    [(0, 0), (0, 1), (0, 2)],  # row 1
    [(1, 0), (1, 1), (1, 2)],  # row 2
    [(2, 0), (2, 1), (2, 2)],  # row 3
    [(0, 0), (1, 0), (2, 0)],  # column 1
    [(0, 1), (1, 1), (2, 1)],  # column 2
    [(0, 2), (1, 2), (2, 2)],  # column 3
    [(0, 0), (1, 1), (2, 2)],  # diagonal 1
    [(2, 0), (1, 1), (0, 2)]   # diagonal 2
]

# Precompute the values for each line
line_values = []
for line in lines:
    values = [grid[i][j] for (i, j) in line]
    line_values.append(values)

# Generate all possible permutations of the 9 cells
all_cells = [(i, j) for i in range(3) for j in range(3)]
total_permutations = 0
valid_permutations = 0

# Since 9! is 362880, it's manageable
for perm in itertools.permutations(all_cells):
    total_permutations += 1
    # Check for each line if it causes disappointment
    disappointed = False
    for line in lines:
        # Get the order of the cells in the line as they appear in the permutation
        order_in_perm = []
        for cell in line:
            order_in_perm.append(perm.index(cell))
        # Sort the cells in the line based on their order in the permutation
        sorted_line = sorted(line, key=lambda x: order_in_perm[line.index(x)])
        # Get the values in the order they are seen
        seen_values = [grid[i][j] for (i, j) in sorted_line]
        # Check if the first two are the same and the third is different
        if seen_values[0] == seen_values[1] and seen_values[1] != seen_values[2]:
            disappointed = True
            break
    if not disappointed:
        valid_permutations += 1

# Calculate the probability
probability = valid_permutations / total_permutations
# Print with high precision
print("{0:.20f}".format(probability))