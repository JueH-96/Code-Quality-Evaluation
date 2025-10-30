import itertools

# Read input and flatten into a list of 9 elements
c_grid = []
for _ in range(3):
    c_grid += list(map(int, input().split()))

# Define all 8 lines in terms of cell indices (0-8)
lines = [
    [0, 1, 2],    # Row 0
    [3, 4, 5],    # Row 1
    [6, 7, 8],    # Row 2
    [0, 3, 6],    # Column 0
    [1, 4, 7],    # Column 1
    [2, 5, 8],    # Column 2
    [0, 4, 8],    # Diagonal 1
    [2, 4, 6]     # Diagonal 2
]

valid = 0
total_permutations = 362880  # 9!

# Generate all permutations of the 9 cells (indices 0-8)
for perm in itertools.permutations(range(9)):
    # Create a position array where pos[cell] is the index in the permutation
    pos = [0] * 9
    for idx, cell in enumerate(perm):
        pos[cell] = idx
    
    invalid = False
    # Check each line
    for line in lines:
        a, b, c = line
        # Get their positions in the permutation
        pa, pb, pc = pos[a], pos[b], pos[c]
        # Create list of tuples (position, cell) and sort by position
        positions = [(pa, a), (pb, b), (pc, c)]
        positions.sort()
        # Extract the three cells in the order they were viewed
        first_cell = positions[0][1]
        second_cell = positions[1][1]
        third_cell = positions[2][1]
        # Check if first two are the same and third is different
        if c_grid[first_cell] == c_grid[second_cell] and c_grid[first_cell] != c_grid[third_cell]:
            invalid = True
            break
    if not invalid:
        valid += 1

# Calculate the probability
probability = valid / total_permutations
# Print with sufficient precision
print("{0:.12f}".format(probability))