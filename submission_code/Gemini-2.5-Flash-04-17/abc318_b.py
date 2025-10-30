# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Define the grid size.
# The coordinates are between 0 and 100.
# A rectangle [A, B] x [C, D] covers unit squares (x, y) to (x+1, y+1)
# where A <= x < B and C <= y < D.
# The possible integer values for x range from 0 up to 99 (since B_i <= 100, max B_i-1 is 99).
# The possible integer values for y range from 0 up to 99 (since D_i <= 100, max D_i-1 is 99).
# A grid indexed from 0 to 99 for both x and y is sufficient.
grid_size = 100
covered = [[False for _ in range(grid_size)] for _ in range(grid_size)]

# Process each rectangle
for _ in range(N):
    # Read rectangle coordinates
    line = sys.stdin.readline().split()
    A, B, C, D = map(int, line)

    # Mark the unit squares covered by this rectangle
    # Iterate through the x and y coordinates of the bottom-left corner of the unit squares.
    # The x-range is [A, B), which in integer steps is A, A+1, ..., B-1.
    # The y-range is [C, D), which in integer steps is C, C+1, ..., D-1.
    for x in range(A, B):
        for y in range(C, D):
            # Mark the unit square (x, y) to (x+1, y+1) as covered.
            # The constraints guarantee that A, C >= 0 and B, D <= 100.
            # Therefore, x (ranging from A to B-1) will be between 0 and 99.
            # Similarly, y (ranging from C to D-1) will be between 0 and 99.
            # The grid indices [x][y] are always valid.
            covered[x][y] = True

# Calculate the total area by counting the number of covered unit squares
total_area = 0
# Iterate through all possible unit squares in the 100x100 region [0, 100] x [0, 100].
for x in range(grid_size):
    for y in range(grid_size):
        if covered[x][y]:
            total_area += 1 # Each covered unit square has an area of 1.

# Print the total area
print(total_area)