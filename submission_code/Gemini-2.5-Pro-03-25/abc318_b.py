# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input, calculates the area covered by the union of rectangles using a grid-based approach,
    and prints the result.
    """
    # Read N, the number of rectangles
    n = int(sys.stdin.readline())

    # Define the maximum coordinate value based on constraints B_i, D_i <= 100.
    # The grid will represent the plane for x and y coordinates from 0 up to 100.
    # Since grid cells represent unit squares [x, x+1) x [y, y+1), we need indices
    # up to 99 for both x and y.
    MAX_COORD = 100
    
    # Initialize a 2D grid (list of lists) representing the coordinate plane.
    # grid[x][y] will be 1 if the unit square starting at (x, y), i.e.,
    # the region x <= real_x < x+1 and y <= real_y < y+1,
    # is covered by at least one rectangle. Otherwise, it's 0.
    # The grid dimensions are MAX_COORD x MAX_COORD, covering indices 0..99 for x and y.
    grid = [[0 for _ in range(MAX_COORD)] for _ in range(MAX_COORD)]

    # Process each of the N rectangles
    for _ in range(n):
        # Read the coordinates of the i-th rectangle: A_i, B_i, C_i, D_i
        # The rectangle covers the region A_i <= x <= B_i and C_i <= y <= D_i
        a, b, c, d = map(int, sys.stdin.readline().split())
        
        # Mark the grid cells covered by this rectangle.
        # A grid cell represented by indices (x_idx, y_idx) corresponds to the unit square area
        # defined by x_idx <= real_x < x_idx + 1 and y_idx <= real_y < y_idx + 1.
        # This unit square is covered by the rectangle if its coordinate range is fully contained
        # within the rectangle's coordinate range.
        # This condition holds if:
        # A_i <= x_idx   AND   x_idx + 1 <= B_i  =>  A_i <= x_idx <= B_i - 1
        # C_i <= y_idx   AND   y_idx + 1 <= D_i  =>  C_i <= y_idx <= D_i - 1
        # Therefore, we iterate through grid indices x_idx from A_i to B_i-1
        # and grid indices y_idx from C_i to D_i-1.
        
        # Loop through the relevant x-indices of the grid
        for x_idx in range(a, b): 
            # Constraints 0 <= a < b <= 100 ensure x_idx is in [0, 99], valid grid indices.
            
            # Loop through the relevant y-indices of the grid
            for y_idx in range(c, d): 
                # Constraints 0 <= c < d <= 100 ensure y_idx is in [0, 99], valid grid indices.
                
                # Mark this grid cell as covered. Setting it to 1 ensures that
                # even if multiple rectangles cover this cell, it's only counted once
                # in the final sum.
                grid[x_idx][y_idx] = 1

    # Calculate the total covered area
    total_area = 0
    # Iterate through all cells in the grid
    for x_idx in range(MAX_COORD):
        for y_idx in range(MAX_COORD):
            # Each cell grid[x_idx][y_idx] marked as 1 represents a unit area (1x1)
            # that is covered by at least one rectangle. Summing these values gives
            # the total area of the union of all rectangles.
            total_area += grid[x_idx][y_idx]

    # Print the final calculated area S as an integer.
    print(total_area)

# Execute the solve function when the script is run
solve()