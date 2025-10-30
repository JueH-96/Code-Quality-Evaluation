import sys

def solve():
    """
    This function encapsulates the entire solution.
    It reads input, performs pre-computation, processes queries, and prints output.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read problem parameters
    try:
        line = input()
        if not line:
            return
        N, Q = map(int, line.split())
    except (IOError, ValueError):
        # Handle cases where input might be empty
        return

    # Read the N x N pattern grid
    P = [input().strip() for _ in range(N)]

    # --- Pre-computation Step ---
    # We use a 2D prefix sum array to answer queries about the number of black
    # squares in any sub-rectangle of the pattern in O(1) time.
    # `ps[i][j]` stores the number of black squares in the rectangle from
    # pattern-coordinate (0, 0) to (i-1, j-1).
    # The array is (N+1)x(N+1) to simplify the calculation with 1-based indexing.
    ps = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            is_black = 1 if P[i][j] == 'B' else 0
            # Standard 2D prefix sum formula
            ps[i + 1][j + 1] = is_black + ps[i][j + 1] + ps[i + 1][j] - ps[i][j]

    def calc(r, c):
        """
        Calculates the total number of black squares in the infinite grid within the
        rectangle from grid-coordinate (0, 0) to (r, c) inclusive.
        """
        if r < 0 or c < 0:
            return 0

        # We work with counts of rows/columns, so r+1 and c+1
        num_rows = r + 1
        num_cols = c + 1

        # Decompose into full blocks and remainder part
        num_full_row_blocks = num_rows // N
        rem_rows = num_rows % N
        
        num_full_col_blocks = num_cols // N
        rem_cols = num_cols % N
        
        # The total count is the sum of black squares in four disjoint regions
        # which together form the rectangle [0,r]x[0,c]. The counts for these
        # regions are calculated using the precomputed `ps` table.
        # `ps[x][y]` gives the count in pattern-rectangle [0, x-1] x [0, y-1].
        
        count_full_blocks = num_full_row_blocks * num_full_col_blocks * ps[N][N]
        count_right_strip = num_full_row_blocks * ps[N][rem_cols]
        count_bottom_strip = num_full_col_blocks * ps[rem_rows][N]
        count_corner_block = ps[rem_rows][rem_cols]
        
        total_black = count_full_blocks + count_right_strip + count_bottom_strip + count_corner_block
        
        return total_black

    # --- Query Processing Step ---
    answers = []
    for _ in range(Q):
        A, B, C, D = map(int, input().split())
        
        # To find the count in a rectangle (A,B) to (C,D), we use the
        # principle of inclusion-exclusion with our `calc` function.
        # Let f(r,c) be calc(r,c).
        # Answer = f(C,D) - f(A-1,D) - f(C,B-1) + f(A-1,B-1)
        result = calc(C, D) - calc(A - 1, D) - calc(C, B - 1) + calc(A - 1, B - 1)
        answers.append(str(result))
    
    # Print all answers at once, separated by newlines, for efficiency.
    print("
".join(answers))

# The problem is structured as a single test case.
solve()