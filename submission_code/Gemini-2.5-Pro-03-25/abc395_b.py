import sys

def solve():
    # Read the integer N from standard input
    n = int(sys.stdin.readline())

    # Create an N x N grid (list of lists) initialized with empty strings
    # This grid will store the final pattern characters ('#' or '.')
    grid = [['' for _ in range(n)] for _ in range(n)]

    # Iterate through each cell of the grid using 0-based indexing.
    # r_idx represents the row index (0 to n-1)
    # c_idx represents the column index (0 to n-1)
    for r_idx in range(n):
        for c_idx in range(n):
            # Calculate the distance of the current cell (r_idx, c_idx)
            # from each of the four borders of the N x N grid.
            
            # Distance from the top border (row 0)
            dist_top = r_idx
            # Distance from the left border (column 0)
            dist_left = c_idx
            # Distance from the bottom border (row n-1)
            dist_bottom = n - 1 - r_idx
            # Distance from the right border (column n-1)
            dist_right = n - 1 - c_idx

            # The minimum of these four distances tells us which "layer" or "frame"
            # the cell belongs to. The outermost layer has min_dist = 0,
            # the next layer inside has min_dist = 1, and so on.
            min_dist = min(dist_top, dist_left, dist_bottom, dist_right)

            # Determine the final color of the cell based on the parity of min_dist.
            # The problem describes a process where nested rectangular regions (frames)
            # are colored starting from the outermost (i=1).
            # Frame i=1 (step 1) colors the region corresponding to min_dist = 0 with black ('#') because i=1 is odd.
            # Frame i=2 (step 2) colors the region corresponding to min_dist = 1 with white ('.') because i=2 is even.
            # Frame i=3 (step 3) colors the region corresponding to min_dist = 2 with black ('#') because i=3 is odd.
            # In general, the color is determined by the parity of the step index 'i'.
            # Since min_dist = i - 1, the parity of min_dist is opposite to the parity of i.
            # If min_dist is even, i = min_dist + 1 is odd, so the color is black ('#').
            # If min_dist is odd, i = min_dist + 1 is even, so the color is white ('.').
            
            if min_dist % 2 == 0:
                # If the minimum distance is even, the cell belongs to a frame
                # painted black ('#').
                grid[r_idx][c_idx] = '#' 
            else:
                # If the minimum distance is odd, the cell belongs to a frame
                # painted white ('.').
                grid[r_idx][c_idx] = '.'

    # Print the resulting grid pattern row by row.
    for r in range(n):
        # For each row, join the characters in the list into a single string
        # and print it to standard output.
        print("".join(grid[r]))

# Call the solve function to execute the program logic.
solve()