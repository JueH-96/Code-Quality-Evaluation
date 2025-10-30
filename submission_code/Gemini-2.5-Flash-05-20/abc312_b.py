import sys

def is_tak_code(grid, r_start, c_start):
    """
    Checks if the 9x9 region starting at (r_start, c_start)
    (0-indexed) satisfies the conditions of a TaK Code.
    """

    # Condition 2: Check black cells
    # Top-left 3x3 region: (r_start+i, c_start+j) for 0 <= i, j <= 2
    for i in range(3):
        for j in range(3):
            if grid[r_start + i][c_start + j] == '.':
                return False

    # Bottom-right 3x3 region: (r_start+i, c_start+j) for 6 <= i, j <= 8
    for i in range(6, 9):
        for j in range(6, 9):
            if grid[r_start + i][c_start + j] == '.':
                return False

    # Condition 3: Check white cells
    # Cells adjacent to top-left 3x3 region
    # Column c_start+3, rows r_start to r_start+3
    for i in range(4): # i from 0 to 3
        if grid[r_start + i][c_start + 3] == '#':
            return False
    # Row r_start+3, columns c_start to c_start+2
    for j in range(3): # j from 0 to 2
        if grid[r_start + 3][c_start + j] == '#':
            return False

    # Cells adjacent to bottom-right 3x3 region
    # Column c_start+5, rows r_start+5 to r_start+8
    for i in range(5, 9): # i from 5 to 8
        if grid[r_start + i][c_start + 5] == '#':
            return False
    # Row r_start+5, columns c_start+6 to c_start+8
    for j in range(6, 9): # j from 6 to 8
        if grid[r_start + 5][c_start + j] == '#':
            return False

    return True

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(N)]

    found_codes = []

    # Iterate through all possible top-left corners of a 9x9 region
    # A 9x9 region starting at (r, c) will end at (r+8, c+8)
    # So, r must be <= N-9 and c must be <= M-9
    for r in range(N - 8): # r goes from 0 to N-9
        for c in range(M - 8): # c goes from 0 to M-9
            if is_tak_code(S, r, c):
                # If it's a TaK Code, add its 1-indexed coordinates
                found_codes.append((r + 1, c + 1))
    
    # Print the results in lexicographical ascending order (already sorted by loops)
    for r_out, c_out in found_codes:
        sys.stdout.write(f"{r_out} {c_out}
")

# Call the solve function to run the program
solve()