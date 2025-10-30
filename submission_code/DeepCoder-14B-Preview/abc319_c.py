import itertools

# Read the grid
c = [list(map(int, input().split())) for _ in range(3)]

# Define all the cells in the grid
cells = [(i, j) for i in range(3) for j in range(3)]

# Define all possible lines (rows, columns, diagonals)
lines = [
    # Rows
    [(0,0), (0,1), (0,2)],
    [(1,0), (1,1), (1,2)],
    [(2,0), (2,1), (2,2)],
    # Columns
    [(0,0), (1,0), (2,0)],
    [(0,1), (1,1), (2,1)],
    [(0,2), (1,2), (2,2)],
    # Diagonals
    [(0,0), (1,1), (2,2)],
    [(2,0), (1,1), (0,2)],
]

valid = 0
total = 0

# Generate all possible permutations of the cells
for perm in itertools.permutations(cells):
    total += 1
    # Create a position map: cell -> its index in the permutation
    pos = {cell: idx for idx, cell in enumerate(perm)}
    is_valid = True
    # Check each line
    for line in lines:
        # Get the indices of the cells in this line within the permutation
        indices = [pos[cell] for cell in line]
        # Sort the indices to determine the order in the permutation
        sorted_indices = sorted(indices)
        # Get the first two cells in this order
        cell1 = perm[sorted_indices[0]]
        cell2 = perm[sorted_indices[1]]
        # Check if their values are the same
        if c[cell1[0]][cell1[1]] == c[cell2[0]][cell2[1]]:
            is_valid = False
            break
    if is_valid:
        valid += 1

# Calculate the probability
probability = valid / total if total != 0 else 0.0

# Print the probability with sufficient precision
print("{0:.20f}".format(probability))