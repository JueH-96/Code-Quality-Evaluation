import sys
from itertools import permutations

# Read input
grid = [list(map(int, input().split())) for _ in range(3)]

# Function to check if a line is disappointed
def is_disappointed(line):
    return line[0] == line[1] and line[2] != line[0]

# Generate all possible permutations of the grid
all_permutations = permutations([num for row in grid for num in row])

# Count the number of valid permutations
valid_count = 0
total_count = 0

for perm in all_permutations:
    total_count += 1
    # Check rows
    for i in range(3):
        if is_disappointed(perm[i*3:(i+1)*3]):
            break
    else:
        # Check columns
        for j in range(3):
            if is_disappointed(perm[j::3]):
                break
        else:
            # Check diagonals
            if is_disappointed((perm[0], perm[4], perm[8])) or is_disappointed((perm[2], perm[4], perm[6])):
                continue
            valid_count += 1

# Calculate probability
probability = valid_count / total_count

# Print the result
print(probability)