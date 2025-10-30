# YOUR CODE HERE
import sys

def solve():
    # Read N and M from the first line of standard input.
    # N: Grid size N x N (1 <= N <= 10^9)
    # M: Number of existing pieces (1 <= M <= 2*10^5)
    line = sys.stdin.readline().split()
    N = int(line[0])
    M = int(line[1])

    # Use a set to store all squares that are unsafe to place a new piece on.
    # An unsafe square is either:
    # 1. A square where an existing piece is located.
    # 2. A square that can be captured by at least one existing piece.
    # A set is used because it automatically handles unique elements, avoiding duplicates.
    unsafe_squares = set()

    # Store the initial piece locations in a list. This list is needed
    # to iterate through each existing piece *exactly once* to calculate
    # its corresponding capture squares. We should iterate over the initial
    # M pieces, not the growing set of unsafe squares.
    pieces = []

    # Read the coordinates (a_k, b_k) of the M existing pieces.
    # Coordinates are 1-indexed: 1 <= a_k, b_k <= N.
    for _ in range(M):
        line = sys.stdin.readline().split()
        r = int(line[0]) # Row coordinate
        c = int(line[1]) # Column coordinate
        piece_location = (r, c) # Store as a tuple (immutable, hashable for set)

        # Add the piece's location to the set of unsafe squares. These squares
        # are by definition not empty and cannot be used for placing a new piece.
        unsafe_squares.add(piece_location)
        # Also store the location in the list so we can process its capture squares.
        pieces.append(piece_location)

    # Define the 8 possible relative moves for a knight-like piece.
    # For a piece at (r, c), a square at (r + dr[i], c + dc[i]) is capturable.
    dr = [-2, -2, -1, -1, +1, +1, +2, +2]
    dc = [-1, +1, -2, +2, -2, +2, -1, +1]

    # Iterate through each existing piece location stored in the 'pieces' list.
    for r, c in pieces:
        # For each piece, calculate the coordinates of the 8 potential capture squares.
        for i in range(8):
            nr = r + dr[i] # Potential capture square row
            nc = c + dc[i] # Potential capture square column

            # Check if the potential capture square (nr, nc) is within the grid boundaries.
            # The grid is defined by 1 <= row <= N and 1 <= column <= N.
            if 1 <= nr <= N and 1 <= nc <= N:
                # If the square is within bounds, it is a capturable square.
                # Add this capturable square's coordinate tuple to the set of unsafe squares.
                # The set will automatically handle cases where multiple pieces
                # can capture the same square.
                unsafe_squares.add((nr, nc))

    # The total number of unique unsafe squares is the size of the 'unsafe_squares' set.
    # This set contains the union of initial piece locations (S) and all unique capturable squares (C) within the grid.
    # i.e., |S union C|
    num_unsafe = len(unsafe_squares)

    # The total number of squares in the N x N grid is N * N.
    # In Python 3, the built-in `int` type can handle arbitrarily large integer values,
    # so N*N will be correctly computed even for N=10^9 (resulting in 10^18).
    total_squares = N * N

    # The number of safe squares where we can place a new piece is the total number of squares
    # in the grid minus the number of unsafe squares. These safe squares are precisely
    # the empty squares (not in S) that are not capturable (not in C).
    # This is equivalent to N*N - |S union C|.
    num_safe = total_squares - num_unsafe

    # Print the final calculated number of safe squares to standard output.
    print(num_safe)

# Call the solve function to execute the program.
solve()