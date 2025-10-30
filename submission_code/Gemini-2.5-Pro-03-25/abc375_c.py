# YOUR CODE HERE
import sys

# Function to read input, solve the problem, and print output
def solve():
    # Read the size of the grid N from standard input
    # N is guaranteed to be an even number.
    N = int(sys.stdin.readline())
    
    # Read the initial grid A as a list of strings.
    # Each string represents a row of the grid.
    # A[i] is the string for the i-th row (0-based index).
    A = [sys.stdin.readline().strip() for _ in range(N)]
    
    # Create the result grid B, initialized as a list of lists of empty strings.
    # B will store the final state of the grid after all operations.
    # Using a list of lists allows modifying individual cells.
    B = [['' for _ in range(N)] for _ in range(N)]

    # Iterate through each cell (r, c) of the final grid B
    # using 0-based indexing (0 <= r < N, 0 <= c < N).
    # r represents the row index, c represents the column index.
    for r in range(N):
        for c in range(N):
            # Calculate the 'layer' index k for the cell (r, c).
            # The layer index determines how many operations affect this cell.
            # An operation i affects cells within the square subgrid defined by
            # rows and columns from i to N+1-i (1-based indexing).
            # A cell (p, q) in 1-based indexing is affected by operations 1 through k,
            # where k = min(p, N+1-p, q, N+1-q).
            # This k represents the minimum distance of the cell to any boundary.
            # We convert this logic to 0-based indexing (r, c) where p=r+1, q=c+1:
            # k = min(r+1, N+1-(r+1), c+1, N+1-(c+1))
            # Simplified: k = min(r+1, N-r, c+1, N-c)
            k = min(r + 1, N - r, c + 1, N - c)
            
            # Determine the coordinates (rk, ck) of the cell in the initial grid A
            # whose color will eventually end up at cell (r, c) in the final grid B.
            # Each operation i effectively applies an inverse transformation to find the source cell.
            # The overall effect of k operations is applying the inverse transformation k times.
            # The transformation rule A^{(i)}_{y, N+1-x} = A^{(i-1)}_{x, y} means the color at (p, q) 
            # in the final grid came from cell (T^{-1})^k(p, q) in the initial grid, where T^{-1}
            # is the inverse transformation.
            # In 0-based indexing, the inverse transformation is T_0^{-1}(r, c) = (N-1-c, r),
            # which corresponds to a 90-degree counter-clockwise rotation about the grid center.
            # We need to compute the coordinates after k applications of T_0^{-1}.
            # The result depends on k modulo 4, as T_0^{-1} has order 4.
            
            rem = k % 4
            
            # Calculate the source coordinates (rk, ck) based on the remainder k % 4
            if rem == 0:
                # If k is a multiple of 4, k applications of T_0^{-1} result in the identity transformation.
                # The source coordinates are the same as the target coordinates (r, c).
                rk, ck = r, c
            elif rem == 1:
                # If k % 4 == 1, one application of T_0^{-1}.
                # Source coordinates are T_0^{-1}(r, c) = (N-1-c, r).
                rk = N - 1 - c
                ck = r
            elif rem == 2:
                # If k % 4 == 2, two applications of T_0^{-1}.
                # Source coordinates are (T_0^{-1})^2(r, c) = (N-1-r, N-1-c).
                rk = N - 1 - r
                ck = N - 1 - c
            else: # rem == 3
                # If k % 4 == 3, three applications of T_0^{-1}.
                # Source coordinates are (T_0^{-1})^3(r, c) = (c, N-1-r).
                rk = c
                ck = N - 1 - r
            
            # Assign the color from the calculated source cell A[rk][ck] 
            # to the current cell B[r][c] in the final grid.
            # A[rk] gets the string for row rk. A[rk][ck] gets the character at column ck.
            B[r][c] = A[rk][ck]

    # After filling the grid B with the final colors, print it row by row.
    for r in range(N):
        # Join the characters in the list B[r] to form a single string representing the row
        # and print it to standard output, followed by a newline.
        print("".join(B[r]))

# Call the solve function to execute the entire process:
# read input, compute the final grid state, and print the output.
solve()