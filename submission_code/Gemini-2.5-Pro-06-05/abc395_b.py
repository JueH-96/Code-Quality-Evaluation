# YOUR CODE HERE
import sys

def solve():
    """
    Reads an integer N and generates the specified N x N pattern.
    """
    try:
        # Read the integer N from standard input.
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Exit if input is not a valid integer or empty.
        return

    # Iterate through each row of the grid using 0-based indexing.
    for row_idx in range(N):
        # Create a list to build the characters for the current row.
        row_chars = []
        
        # Iterate through each column of the grid using 0-based indexing.
        for col_idx in range(N):
            # Convert 0-based indices to 1-based (r, c) for calculation,
            # as the problem's formula uses 1-based coordinates.
            r = row_idx + 1
            c = col_idx + 1

            # Determine the index 'i' of the last operation that colored the
            # cell (r, c). This is the minimum of the cell's 1-based distances
            # from the four edges of the N x N grid.
            last_op_index = min(r, c, N + 1 - r, N + 1 - c)

            # The color is black ('#') if the operation index is odd,
            # and white ('.') if it is even.
            if last_op_index % 2 == 1:
                row_chars.append('#')
            else:
                row_chars.append('.')
        
        # Join the characters to form the row's string and print to standard output.
        print("".join(row_chars))

# Execute the solution.
solve()