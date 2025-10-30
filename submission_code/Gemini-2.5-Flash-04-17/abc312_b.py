# YOUR CODE HERE
import sys

# Read N and M
line = sys.stdin.readline().split()
N = int(line[0])
M = int(line[1])

# Read the grid
grid = []
for _ in range(N):
    grid.append(sys.stdin.readline().strip())

# Function to check if a 9x9 region starting at (r, c) (0-indexed) is a TaK Code
def is_tak_code(grid, r, c, N, M):
    # Check bounds: the 9x9 region must be completely inside the N x M grid
    # The region covers rows r to r+8 and columns c to c+8
    if r + 8 >= N or c + 8 >= M:
        return False

    # Check TL 3x3 black (relative rows 0-2, cols 0-2 within the 9x9 region)
    # Absolute coordinates: rows r to r+2, cols c to c+2
    for ri in range(3):
        for cj in range(3):
            if grid[r + ri][c + cj] == '.':
                return False

    # Check BR 3x3 black (relative rows 6-8, cols 6-8 within the 9x9 region)
    # Absolute coordinates: rows r+6 to r+8, cols c+6 to c+8
    for ri in range(6, 9):
        for cj in range(6, 9):
            if grid[r + ri][c + cj] == '.':
                return False

    # Check white cells adjacent to TL 3x3 region
    # These are cells (ri', cj') within the 9x9 region [0,8]x[0,8]
    # such that max(|ri'-ri|, |cj'-cj|)==1 for some (ri,cj) in [0,2]x[0,2],
    # and (ri', cj') is not in [0,2]x[0,2].
    # The relative coordinates are {(0,3), (1,3), (2,3), (3,3)} and {(3,0), (3,1), (3,2)}
    # Absolute coordinates:
    # (r+ri, c+3) for ri in [0, 3]
    for ri in range(4):
        if grid[r + ri][c + 3] == '#':
            return False
    # (r+3, c+cj) for cj in [0, 2]
    for cj in range(3):
        if grid[r + 3][c + cj] == '#':
            return False

    # Check white cells adjacent to BR 3x3 region
    # These are cells (ri', cj') within the 9x9 region [0,8]x[0,8]
    # such that max(|ri'-ri|, |cj'-cj|)==1 for some (ri,cj) in [6,8]x[6,8],
    # and (ri', cj') is not in [6,8]x[6,8].
    # The relative coordinates are {(5,5), (5,6), (5,7), (5,8)} and {(6,5), (7,5), (8,5)}
    # Absolute coordinates:
    # (r+5, c+cj) for cj in [5, 8]
    for cj in range(5, 9):
        if grid[r + 5][c + cj] == '#':
            return False
    # (r+ri, c+5) for ri in [6, 8]
    for ri in range(6, 9):
        if grid[r + ri][c + 5] == '#':
            return False

    # If all checks pass
    return True

# Iterate through all possible top-left corners (r, c) (0-indexed)
# The top-left corner (r, c) of a 9x9 region means rows r to r+8 and columns c to c+8.
# For this to be within the N x M grid, we need r+8 < N and c+8 < M.
# This means r <= N-9 and c <= M-9.
# Possible r values: 0 to N-9
# Possible c values: 0 to M-9
# range(N - 8) covers indices 0, 1, ..., N-9.
# range(M - 8) covers indices 0, 1, ..., M-9.
for r in range(N - 8):
    for c in range(M - 8):
        if is_tak_code(grid, r, c, N, M):
            # Output is 1-indexed row and column
            print(r + 1, c + 1)