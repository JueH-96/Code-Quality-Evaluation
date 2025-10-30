import sys
import itertools

def calculate_probability(grid):
    # Generate all possible permutations of the grid
    perms = list(itertools.permutations(range(9)))

    # Initialize count of valid permutations
    valid_perms = 0

    # Iterate over each permutation
    for perm in perms:
        # Initialize a flag to indicate if the permutation is valid
        is_valid = True

        # Iterate over each possible line (horizontal, vertical, diagonal)
        for line in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            # Get the indices of the line in the permutation
            idx1, idx2, idx3 = [perm.index(i) for i in line]

            # Check if the first two elements are the same and the third is different
            if idx1 < idx2 and idx2 < idx3 and grid[line[0] // 3][line[0] % 3] == grid[line[1] // 3][line[1] % 3] and grid[line[0] // 3][line[0] % 3] != grid[line[2] // 3][line[2] % 3]:
                # If so, mark the permutation as invalid
                is_valid = False
                break

        # If the permutation is valid, increment the count
        if is_valid:
            valid_perms += 1

    # Calculate and return the probability
    return valid_perms / len(perms)

# Read the grid from stdin
grid = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]

# Calculate and print the probability
print(calculate_probability(grid))