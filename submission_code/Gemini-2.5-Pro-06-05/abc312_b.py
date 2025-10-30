import sys

def solve():
    """
    Reads the grid and finds all Tak Code regions.
    """
    # Read grid dimensions and the grid itself
    try:
        N, M = map(int, sys.stdin.readline().split())
        grid = [sys.stdin.readline().strip() for _ in range(N)]
    except (IOError, ValueError):
        # This case is unlikely in a standard contest environment but is good practice
        return

    def is_tak_code(r, c):
        """
        Checks if the 9x9 region starting at (r, c) is a Tak Code.
        r, c are 0-based indices.
        """
        # Condition 1: Top-left 3x3 region must be black ('#')
        for i in range(r, r + 3):
            for j in range(c, c + 3):
                if grid[i][j] != '#':
                    return False

        # Condition 2: Bottom-right 3x3 region must be black ('#')
        for i in range(r + 6, r + 9):
            for j in range(c + 6, c + 9):
                if grid[i][j] != '#':
                    return False

        # Condition 3: 7 cells adjacent to the top-left 3x3 region must be white ('.')
        # This forms an 'L' shape of white cells.
        for j in range(c, c + 4):  # Horizontal part including the corner
            if grid[r + 3][j] != '.':
                return False
        for i in range(r, r + 3):  # Vertical part
            if grid[i][c + 3] != '.':
                return False

        # Condition 4: 7 cells adjacent to the bottom-right 3x3 region must be white ('.')
        # This forms an inverted 'L' shape of white cells.
        for j in range(c + 5, c + 9):  # Horizontal part including the corner
            if grid[r + 5][j] != '.':
                return False
        for i in range(r + 6, r + 9):  # Vertical part
            if grid[i][c + 5] != '.':
                return False
        
        # If all conditions are satisfied, it is a Tak Code
        return True

    # Iterate through all possible top-left corners (i, j) of a 9x9 region.
    # The loop range ensures the 9x9 region is fully contained in the NxM grid.
    for i in range(N - 8):
        for j in range(M - 8):
            if is_tak_code(i, j):
                # The problem asks for 1-based indices.
                print(i + 1, j + 1)

# Execute the solution
solve()