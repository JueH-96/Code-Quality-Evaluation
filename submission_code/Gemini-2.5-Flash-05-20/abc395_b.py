import sys

def solve():
    N = int(sys.stdin.readline())

    # Initialize the N x N grid.
    # We use a list of lists of characters. Since all cells will eventually be colored,
    # the initial values don't strictly matter, but using empty strings or a default
    # character is fine.
    grid = [['' for _ in range(N)] for _ in range(N)]

    # Iterate through the operations, where 'k' is the 0-indexed loop variable.
    # 'i_1_indexed' corresponds to 'i' in the problem statement (1-indexed).
    for k in range(N):
        i_1_indexed = k + 1

        # Calculate 'j_prime' (1-indexed) according to the problem's formula.
        j_prime_1_indexed = N + 1 - i_1_indexed

        # Check the condition: if i <= j_prime, perform the coloring operation.
        # Otherwise, do nothing for this iteration.
        if i_1_indexed <= j_prime_1_indexed:
            # Determine the character to fill based on whether 'i' is odd or even.
            # Odd 'i' means black ('#'), even 'i' means white ('.').
            color_char = '#' if i_1_indexed % 2 != 0 else '.'

            # Determine the 0-indexed coordinates for the top-left and bottom-right
            # corners of the rectangular region.
            # The top-left cell (i,i) (1-indexed) becomes (i-1, i-1) (0-indexed).
            row_start = i_1_indexed - 1
            col_start = i_1_indexed - 1
            # The bottom-right cell (j_prime, j_prime) (1-indexed) becomes (j_prime-1, j_prime-1) (0-indexed).
            row_end = j_prime_1_indexed - 1
            col_end = j_prime_1_indexed - 1

            # Fill the rectangular region with the determined color.
            # This loop naturally handles overwriting previous colors.
            for r in range(row_start, row_end + 1):
                for c in range(col_start, col_end + 1):
                    grid[r][c] = color_char
        # If i > j_prime, the 'else' branch is implicitly 'do nothing',
        # which matches the problem specification.

    # After all operations, print the final state of the grid.
    # Each row is joined into a string and printed, followed by a newline.
    for r in range(N):
        sys.stdout.write("".join(grid[r]) + "
")

# Call the solve function to execute the program.
solve()