import sys

def solve():
    # Read the number of rectangular sheets.
    N = int(sys.stdin.readline())
    
    # Define the maximum coordinate value based on problem constraints.
    # The problem states 0 <= B_i <= 100 and 0 <= D_i <= 100.
    # A rectangle A_i <= x <= B_i and C_i <= y <= D_i covers unit squares
    # whose bottom-left corners (x_unit, y_unit) satisfy:
    # A_i <= x_unit < B_i and C_i <= y_unit < D_i.
    # This means x_unit can go up to B_i - 1 and y_unit can go up to D_i - 1.
    # Since B_i and D_i can be 100, x_unit and y_unit can go up to 99.
    # Therefore, we need a grid of size 100x100, where indices range from 0 to 99.
    GRID_SIZE = 100 
    
    # Initialize a 2D boolean grid.
    # `covered[x][y]` will be True if the unit square with bottom-left corner (x,y)
    # is covered by at least one sheet. Otherwise, it's False.
    covered = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    # Process each of the N rectangular sheets.
    for _ in range(N):
        # Read the coordinates of the current sheet.
        # A, B, C, D represent (A <= x <= B) and (C <= y <= D).
        A, B, C, D = map(int, sys.stdin.readline().split())
        
        # Iterate over the unit squares covered by the current rectangle.
        # For a rectangle defined by [A, B] on x-axis and [C, D] on y-axis,
        # it covers unit squares [x_idx, x_idx+1) x [y_idx, y_idx+1) where
        # x_idx runs from A to B-1 (inclusive)
        # and y_idx runs from C to D-1 (inclusive).
        for x_idx in range(A, B):
            for y_idx in range(C, D):
                # Mark this unit square as covered.
                covered[x_idx][y_idx] = True
                
    # Calculate the total area by counting the number of True cells in the grid.
    # Each True cell represents one unit square of area.
    total_area = 0
    for x_idx in range(GRID_SIZE):
        for y_idx in range(GRID_SIZE):
            if covered[x_idx][y_idx]:
                total_area += 1
                
    # Print the final total area.
    print(total_area)

# Call the solve function to run the program.
if __name__ == '__main__':
    solve()