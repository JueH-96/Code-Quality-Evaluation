# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read().strip().split()

# Get the size of the grid
N = int(input[0])

# Initialize grids
grid_A = []
grid_B = []

# Populate grid A
for i in range(1, N + 1):
    grid_A.append(input[i])

# Populate grid B
for i in range(N + 1, 2 * N + 1):
    grid_B.append(input[i])

# Find the differing cell
for i in range(N):
    for j in range(N):
        if grid_A[i][j] != grid_B[i][j]:
            print(i + 1, j + 1)
            sys.exit(0)