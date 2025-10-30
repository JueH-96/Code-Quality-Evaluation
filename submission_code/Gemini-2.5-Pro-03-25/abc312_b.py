import sys

def check(grid, r, c):
    """
    Checks if the 9x9 subgrid starting at (r, c) satisfies Tak Code conditions.

    Args:
        grid: The full NxM grid (list of strings).
        r: The starting row index (0-based).
        c: The starting column index (0-based).

    Returns:
        True if it's a Tak Code, False otherwise.
    """
    # Condition 1: Top-left 3x3 region must be black ('#')
    # Check cells grid[i][j] for i in [r, r+1, r+2] and j in [c, c+1, c+2]
    for i in range(r, r + 3):
        for j in range(c, c + 3):
            # The main loop ensures r+2 < N and c+2 < M, so indices are valid
            if grid[i][j] != '#':
                return False

    # Condition 2: Bottom-right 3x3 region must be black ('#')
    # Check cells grid[i][j] for i in [r+6, r+7, r+8] and j in [c+6, c+7, c+8]
    for i in range(r + 6, r + 9):
        for j in range(c + 6, c + 9):
            # The main loop ensures r+8 < N and c+8 < M, so indices are valid
            if grid[i][j] != '#':
                return False

    # Condition 3: Cells adjacent to the top-left 3x3 must be white ('.')
    # These are the 7 cells within the 9x9 grid that border the top-left 3x3.
    # Check row r+3, columns c to c+3 (4 cells: [r+3][c], [r+3][c+1], [r+3][c+2], [r+3][c+3])
    # The main loop ensures r+3 < N and c+3 < M
    for j in range(c, c + 4):
        if grid[r+3][j] != '.':
            return False
    # Check column c+3, rows r to r+2 (3 cells: [r][c+3], [r+1][c+3], [r+2][c+3])
    # (Cell [r+3][c+3] was already checked in the loop above)
    # The main loop ensures r+2 < N and c+3 < M
    for i in range(r, r + 3):
        if grid[i][c+3] != '.':
            return False

    # Condition 4: Cells adjacent to the bottom-right 3x3 must be white ('.')
    # These are the 7 cells within the 9x9 grid that border the bottom-right 3x3.
    # Check row r+5, columns c+5 to c+8 (4 cells: [r+5][c+5], [r+5][c+6], [r+5][c+7], [r+5][c+8])
    # The main loop ensures r+5 < N and c+8 < M
    for j in range(c + 5, c + 9):
         if grid[r+5][j] != '.':
             return False
    # Check column c+5, rows r+6 to r+8 (3 cells: [r+6][c+5], [r+7][c+5], [r+8][c+5])
    # (Cell [r+5][c+5] was already checked in the loop above)
    # The main loop ensures r+8 < N and c+5 < M
    for i in range(r + 6, r + 9):
         if grid[i][c+5] != '.':
             return False

    # If all conditions are met
    return True


def solve():
    """
    Reads input, finds Tak Codes, and prints the results.
    """
    # Read grid dimensions
    n, m = map(int, sys.stdin.readline().split())

    # Read grid content
    grid = [sys.stdin.readline().strip() for _ in range(n)]

    # List to store the top-left coordinates of found Tak Codes
    results = []

    # Iterate through all possible top-left corners (r, c) for a 9x9 region.
    # The range ends at n-9 and m-9 respectively (so r goes from 0 to n-9,
    # c goes from 0 to m-9). This ensures the 9x9 grid starting at (r, c)
    # fits entirely within the NxM grid (maximum index accessed is r+8 and c+8).
    for r in range(n - 8):
        for c in range(m - 8):
            # Check if the 9x9 subgrid starting at (r, c) is a Tak Code
            if check(grid, r, c):
                # Store 1-based index as required by the output format
                results.append((r + 1, c + 1))

    # Print results. The results list will be naturally sorted lexicographically
    # (first by row 'r', then by column 'c') because of the nested loop order.
    for r_res, c_res in results:
        print(f"{r_res} {c_res}")

# Execute the solver function when the script is run
if __name__ == '__main__':
    solve()