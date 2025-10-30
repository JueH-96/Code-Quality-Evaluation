import itertools
import sys
import math

# Read input from stdin
data = sys.stdin.read().split()
num = [int(x) for x in data]  # List of nine numbers in row-major order

# Define the eight lines (rows, columns, diagonals) with their indices
lines_def = [
    (0, 1, 2),  # Row 1
    (3, 4, 5),  # Row 2
    (6, 7, 8),  # Row 3
    (0, 3, 6),  # Column 1
    (1, 4, 7),  # Column 2
    (2, 5, 8),  # Column 3
    (0, 4, 8),  # Main diagonal
    (2, 4, 6)   # Anti-diagonal
]

# Find all constrained lines where exactly two cells have the same number
constraints = []
for idxs in lines_def:
    val0 = num[idxs[0]]
    val1 = num[idxs[1]]
    val2 = num[idxs[2]]
    # Check if exactly two values are the same
    if (val0 == val1 and val0 != val2) or (val0 == val2 and val0 != val1) or (val1 == val2 and val1 != val0):
        # Find the index of the different cell
        if val0 != val1 and val0 != val2:  # val0 is different
            diff_idx = idxs[0]
        elif val1 != val0 and val1 != val2:  # val1 is different
            diff_idx = idxs[1]
        elif val2 != val0 and val2 != val1:  # val2 is different
            diff_idx = idxs[2]
        else:
            # This should not happen due to input constraints, but handle just in case
            continue  # Skip this line, though it shouldn't occur
        constraints.append((idxs, diff_idx))  # Store the trio and the different index

# Total number of permutations
total_perms = math.factorial(9)

# Count the number of good permutations
good_count = 0
for perm in itertools.permutations(range(9)):  # perm is a tuple of cell indices in visit order
    # Create a visit time list for the permutation
    visit_time = [0] * 9  # visit_time[cell] = time of visit
    for t, cell in enumerate(perm):
        visit_time[cell] = t
    
    # Check all constraints
    is_good = True
    for trio, diff_idx in constraints:
        # Find the cell in the trio with the maximum visit time (visited last)
        max_cell = max(trio, key=lambda x: visit_time[x])
        if max_cell == diff_idx:  # If the different cell is last, violation
            is_good = False
            break  # No need to check further constraints
    
    if is_good:
        good_count += 1

# Calculate probability
prob = good_count / total_perms

# Output the probability with high precision
print(f"{prob:.15f}")