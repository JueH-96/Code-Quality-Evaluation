# YOUR CODE HERE
import sys

# Read the size of the grid N from standard input
n = int(sys.stdin.readline())

# Read the N rows of the grid as strings
# Each string represents a row in the grid A.
a_str = [sys.stdin.readline().strip() for _ in range(n)]

# Convert the grid of strings into a 2D list of integers (grid A)
# Each character '0' or '1' is converted to an integer 0 or 1.
grid_a = [[int(c) for c in row] for row in a_str]

# Create a new grid B as a deep copy of grid A.
# We will modify grid B based on the values from the original grid A.
# This approach avoids issues where updating a cell might affect the calculation
# for another cell in the same shift operation if done in-place.
# Using list comprehension `[row[:] for row in grid_a]` creates a shallow copy
# of the outer list but copies the inner lists. Since integers are immutable,
# this works as a deep copy for a 2D list of integers.
grid_b = [row[:] for row in grid_a]

# The problem statement guarantees N >= 2, so there's always an outer layer (boundary) of squares.
# Perform the clockwise shift operation on the outer squares.
# We update the values in grid B based on the corresponding previous values in grid A.
if n >= 2:
    # 1. Shift the top row elements (excluding the top-right corner) one step to the right.
    # The element originally at A[0][j-1] moves to the position B[0][j].
    # This loop covers indices j from 1 to N-1.
    # Example: A[0][0] moves to B[0][1], A[0][1] moves to B[0][2], ..., A[0][N-2] moves to B[0][N-1].
    for j in range(1, n):
        grid_b[0][j] = grid_a[0][j-1]

    # 2. Shift the right column elements (excluding the bottom-right corner) one step down.
    # The element originally at A[i-1][N-1] moves to the position B[i][N-1].
    # This loop covers indices i from 1 to N-1.
    # Example: A[0][N-1] moves to B[1][N-1], A[1][N-1] moves to B[2][N-1], ..., A[N-2][N-1] moves to B[N-1][N-1].
    for i in range(1, n):
        grid_b[i][n-1] = grid_a[i-1][n-1]

    # 3. Shift the bottom row elements (excluding the bottom-left corner) one step to the left.
    # The element originally at A[N-1][j+1] moves to the position B[N-1][j].
    # This loop covers indices j from 0 to N-2.
    # Example: A[N-1][1] moves to B[N-1][0], A[N-1][2] moves to B[N-1][1], ..., A[N-1][N-1] moves to B[N-1][N-2].
    for j in range(n - 1): # Loop `j` from 0 up to (but not including) n-1
        grid_b[n-1][j] = grid_a[n-1][j+1]

    # 4. Shift the left column elements (excluding the top-left corner) one step up.
    # The element originally at A[i+1][0] moves to the position B[i][0].
    # This loop covers indices i from 0 to N-2.
    # Example: A[1][0] moves to B[0][0], A[2][0] moves to B[1][0], ..., A[N-1][0] moves to B[N-2][0].
    for i in range(n - 1): # Loop `i` from 0 up to (but not including) n-1
        grid_b[i][0] = grid_a[i+1][0]

# Print the resulting grid B to standard output.
# Iterate through each row of the modified grid B.
for i in range(n):
    # For each row (which is a list of integers):
    # 1. Convert each integer in the row to its string representation using `map(str, ...)`.
    # 2. Join these strings together without any separator using `"".join(...)`.
    # 3. Print the resulting string for the row.
    print("".join(map(str, grid_b[i])))