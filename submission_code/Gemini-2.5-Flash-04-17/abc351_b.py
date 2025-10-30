# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Read grid A
grid_a = []
for _ in range(N):
    grid_a.append(sys.stdin.readline().strip())

# Read grid B
grid_b = []
for _ in range(N):
    grid_b.append(sys.stdin.readline().strip())

# Find the differing cell
for i in range(N):
    for j in range(N):
        if grid_a[i][j] != grid_b[i][j]:
            # Found the difference
            # Print 1-based indices and exit
            print(i + 1, j + 1)
            sys.exit() # Exit the program immediately after printing

# This part should not be reached based on the problem guarantee
# that exactly one difference exists.