# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Read grid A
A = []
for _ in range(N):
    A.append(list(map(int, list(sys.stdin.readline().strip()))))

# 1. Extract outer values in clockwise order
outer_values = []

# Top row (0, j) for j=0 to N-1
for j in range(N):
    outer_values.append(A[0][j])

# Right column (i, N-1) for i=1 to N-1
for i in range(1, N):
    outer_values.append(A[i][N-1])

# Bottom row (N-1, j) for j=N-2 down to 0
# range(N-2, -1, -1) works correctly for N=2 (range(0, -1, -1) -> [0]) and N>2
for j in range(N - 2, -1, -1):
    outer_values.append(A[N-1][j])

# Left column (i, 0) for i=N-2 down to 1
# range(N-2, 0, -1) works correctly for N=2 (range(0, 0, -1) -> empty) and N>2
for i in range(N - 2, 0, -1):
    outer_values.append(A[i][0])

# 2. Shift the extracted list clockwise by one position
# The last element moves to the front.
shifted_values = [outer_values[-1]] + outer_values[:-1]

# 3. Put shifted values back into the grid
# The order of putting back must match the order of extraction.

k = 0 # index in shifted_values

# Top row (0, j) for j=0 to N-1
for j in range(N):
    A[0][j] = shifted_values[k]
    k += 1

# Right column (i, N-1) for i=1 to N-1
for i in range(1, N):
    A[i][N-1] = shifted_values[k]
    k += 1

# Bottom row (N-1, j) for j=N-2 down to 0
for j in range(N - 2, -1, -1):
    A[N-1][j] = shifted_values[k]
    k += 1

# Left column (i, 0) for i=N-2 down to 1
for i in range(N - 2, 0, -1):
    A[i][0] = shifted_values[k]
    k += 1

# 4. Print the modified grid A
for row in A:
    print("".join(map(str, row)))