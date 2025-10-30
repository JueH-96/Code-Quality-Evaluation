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

# Function to check if a permutation is valid
def is_valid(permutation):
    for line in lines:
        # Get the values in the line in the order of the permutation
        values = [grid[i][j] for (i, j) in line]
        # Get the order in which they are seen
        order = [permutation.index((i, j)) for (i, j) in line]
        # Sort the order to get the sequence in which they are seen
        sorted_order = sorted(range(3), key=lambda x: order[x])
        # Check if the first two are the same and the third is different
        if values[sorted_order[0]] == values[sorted_order[1]] and values[sorted_order[1]] != values[sorted_order[2]]:
            return False
    return True

# Generate all possible permutations of the cells
cells = [(i, j) for i in range(3) for j in range(3)]
total = 0
valid = 0
for perm in itertools.permutations(cells):
    total += 1
    if is_valid(perm):
        valid += 1

# Calculate the probability
probability = valid / total
print("{0:.20f}".format(probability))