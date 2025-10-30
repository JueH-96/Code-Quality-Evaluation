import sys

# It is recommended to wrap the solution in a function for clarity and organization,
# but for some competitive programming platforms, a simple script is sufficient.
# The main logic is presented here as a sequential script.

# Read N and M from the first line of standard input.
# N: the size of the grid (N x N).
# M: the number of existing pieces.
try:
    line = sys.stdin.readline()
    if line:
        N, M = map(int, line.split())
    else:
        # Handle empty input
        sys.exit()
except (IOError, ValueError):
    sys.exit()

# We will store the coordinates of the M pieces.
# `pieces` is a list used for iteration.
# `occupied_squares` is a set for efficient calculation of the union of forbidden squares.
pieces = []
occupied_squares = set()

# Read the coordinates of the M pieces.
for _ in range(M):
    try:
        line = sys.stdin.readline()
        if line:
            a, b = map(int, line.split())
            pieces.append((a, b))
            occupied_squares.add((a, b))
        else:
            break
    except (IOError, ValueError):
        break

# Define the 8 possible knight moves as relative (row, column) offsets.
moves = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

# Create a set to store all unique squares attacked by any of the pieces.
# Using a set automatically handles cases where a square is attacked by multiple pieces.
attacked_squares = set()

# Iterate through each existing piece to determine all squares it attacks.
for r, c in pieces:
    # For each piece, generate its 8 potential attack squares.
    for dr, dc in moves:
        # Calculate the coordinates of the potentially attacked square.
        nr, nc = r + dr, c + dc
        
        # A square is only attacked if it is within the N x N grid boundaries.
        # The grid coordinates are 1-indexed.
        if 1 <= nr <= N and 1 <= nc <= N:
            attacked_squares.add((nr, nc))

# The set of all forbidden squares is the union of squares that are occupied
# and squares that are attacked.
forbidden_squares = occupied_squares.union(attacked_squares)

# The number of forbidden squares is simply the size of this union set.
num_forbidden = len(forbidden_squares)

# The total number of squares on the grid is N*N. Python's integers
# can handle the large values resulting from N up to 10^9.
total_squares = N * N

# The number of safe squares is the total number of squares minus the
# number of forbidden squares.
answer = total_squares - num_forbidden

# Print the final answer to standard output.
print(answer)