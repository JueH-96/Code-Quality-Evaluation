import itertools
import sys
import math

# Read input
grid = []
for _ in range(3):
    grid.append(list(map(int, sys.stdin.readline().split())))

# Map cell index (0-8) to (row, col)
idx_to_coord = {i: (i // 3, i % 3) for i in range(9)}

def get_value(cell_idx):
    r, c = idx_to_coord[cell_idx]
    return grid[r][c]

# Define lines by cell indices
lines = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8), # Horizontal (row 0, 1, 2)
    (0, 3, 6), (1, 4, 7), (2, 5, 8), # Vertical (col 0, 1, 2)
    (0, 4, 8), (2, 4, 6)  # Diagonal (main, anti)
]

# Find all potential disappointment triggers
# A trigger is (cell_idx_A1, cell_idx_A2, cell_idx_B)
# where cells A1 and A2 have the same value V_A, cell B has value V_B, V_A != V_B,
# and {A1, A2, B} form a line.
# Disappointment happens if A1 and A2 are seen before B in the sequence.
triggers = []
for line in lines:
    c0, c1, c2 = line
    v0, v1, v2 = get_value(c0), get_value(c1), get_value(c2)

    # Check for (A, A, B) type patterns. The problem guarantees no (A, A, A) patterns.
    if v0 == v1 and v0 != v2:
        triggers.append((c0, c1, c2))
    if v0 == v2 and v0 != v1:
        triggers.append((c0, c2, c1))
    if v1 == v2 and v1 != v0:
        triggers.append((c1, c2, c0))

# Now, iterate through all permutations of cell indices [0, 1, ..., 8]
count_good_permutations = 0
total_permutations = 0 # Initialize total_permutations

all_cells_indices = list(range(9))

for perm in itertools.permutations(all_cells_indices):
    total_permutations += 1
    
    # Create a mapping from cell index to its position (0-indexed) in the permutation
    # pos[cell_idx] = position_in_perm
    pos = {cell_idx: i for i, cell_idx in enumerate(perm)}
    
    disappointed = False
    for t_same1, t_same2, t_diff in triggers:
        # Check if cell t_same1 and cell t_same2 appear before cell t_diff
        # This means the position of t_same1 in the permutation is less than
        # the position of t_diff, AND the position of t_same2 is less than
        # the position of t_diff.
        if pos[t_same1] < pos[t_diff] and pos[t_same2] < pos[t_diff]:
            disappointed = True
            break # Found a disappointment trigger occurrence, this perm is bad
    
    if not disappointed:
        count_good_permutations += 1

# The probability is the ratio of good permutations to the total number of permutations
# If total_permutations is 0 (should not happen for 9 elements, 9! > 0), probability is 0.
probability = count_good_permutations / total_permutations if total_permutations > 0 else 0.0

# Print the result with required precision
# 10^-8 absolute error requires printing enough digits, e.g., 17 decimal places.
print(f"{probability:.17f}")