# YOUR CODE HERE
import sys

def solve():
    # Read N (grid size) and M (number of pieces) from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Use a set to store all forbidden squares. A square is forbidden if it's either
    # occupied by an initial piece or attacked by an initial piece.
    # Sets provide efficient addition and automatically handle duplicate entries.
    forbidden_squares = set() 

    # Use a list to store the coordinates of the initial pieces.
    # We need this list to iterate through the pieces later and calculate the squares they attack.
    pieces_list = [] 

    # Read the coordinates of the M initial pieces
    for _ in range(M):
        # Read piece coordinates (row, column)
        coords = tuple(map(int, sys.stdin.readline().split()))
        
        # Add the coordinates to the list for later iteration
        pieces_list.append(coords)
        
        # Add the initial piece's position to the set of forbidden squares.
        # The problem statement guarantees that initial piece positions are distinct,
        # so we don't strictly need to check for duplicates before adding,
        # but using a set inherently handles this.
        forbidden_squares.add(coords)

    # Define the 8 possible relative moves for the pieces.
    # These moves are similar to a knight's moves in chess.
    # Each tuple represents (delta_row, delta_column).
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1), # Moves involving +2 row/col change
        (-2, -1), (-1, -2), (1, -2), (2, -1) # Moves involving -2 row/col change
    ]

    # Iterate through each initial piece's location stored in the list
    for r, c in pieces_list: 
        # For each piece at coordinates (r, c), calculate the 8 potential squares it attacks
        for dr, dc in moves:
            # Calculate the absolute coordinates (nr, nc) of the potential attacked square
            # by adding the relative move (dr, dc) to the piece's position (r, c).
            nr = r + dr
            nc = c + dc
            
            # Check if the calculated coordinates (nr, nc) fall within the N x N grid boundaries.
            # The grid coordinates are 1-based, so we check if 1 <= nr <= N and 1 <= nc <= N.
            if 1 <= nr <= N and 1 <= nc <= N:
                # If the square is within the grid boundaries, it is potentially attacked by the piece at (r, c).
                # Add this attacked square's coordinates to the set of forbidden squares.
                # The `set.add` operation is efficient and automatically handles cases where the square
                # is already in the set (e.g., it was an initial piece location, or
                # it was already marked as attacked by another piece). Only unique coordinates are stored.
                forbidden_squares.add((nr, nc))

    # Calculate the total number of squares on the N x N grid.
    # N can be as large as 10^9, so N*N can be up to 10^18. 
    # Python's built-in integers handle arbitrarily large integers, so this calculation is safe from overflow.
    total_squares = N * N 
    
    # The number of squares where a new piece can be safely placed is the
    # total number of squares on the grid minus the number of forbidden squares.
    # A square is forbidden if it is initially occupied OR if it is attacked by any initial piece.
    # The size of the `forbidden_squares` set gives us the count of unique forbidden squares.
    result = total_squares - len(forbidden_squares)
    
    # Print the final calculated result to standard output.
    print(result)

# Call the main function `solve()` to execute the program logic.
solve()
# END OF CODE