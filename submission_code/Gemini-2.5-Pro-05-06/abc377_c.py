import sys

# It's good practice to put the main logic in a function.
def solve():
    # Read N and M from the first line of input.
    # N: grid size (N x N)
    # M: number of existing pieces
    N, M = map(int, sys.stdin.readline().split())

    # Using a set to store coordinates of "forbidden" squares.
    # A square (r, c) is forbidden if:
    # 1. It is occupied by one of the M existing pieces.
    # 2. It can be captured by one of the M existing pieces if you place your piece there.
    # Using a set automatically handles duplicates, so we count unique forbidden squares.
    forbidden_coords = set()

    # These are the 8 relative moves a knight-like piece can make.
    # (dr, dc) represents (change in row, change in column).
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),  # Moves with positive dc or positive dr
        (-2, -1), (-1, -2), (1, -2), (2, -1) # Moves with negative dc or negative dr
    ]

    # Process each of the M existing pieces.
    for _ in range(M):
        # Read the coordinates (r, c) of an existing piece.
        # The problem uses 1-indexed coordinates.
        r, c = map(int, sys.stdin.readline().split())
        
        # Add the square occupied by the current piece to the set of forbidden squares.
        # (a_k, b_k) are guaranteed to be within [1, N] as per problem constraints.
        forbidden_coords.add((r, c))
        
        # Determine all squares attacked by the current piece at (r, c).
        # Add these to the set of forbidden squares if they are on the grid.
        for dr, dc in knight_moves:
            attacked_r, attacked_c = r + dr, c + dc
            
            # Check if the attacked square (attacked_r, attacked_c) is within the grid boundaries.
            # The grid is N x N, with rows from 1 to N and columns from 1 to N.
            if 1 <= attacked_r <= N and 1 <= attacked_c <= N:
                forbidden_coords.add((attacked_r, attacked_c))
                
    # The number of unique forbidden squares.
    num_forbidden_squares = len(forbidden_coords)
    
    # Total number of squares on the N x N grid.
    # N can be up to 10^9, so N*N can be up to 10^18.
    # Python's integers handle arbitrary precision, so N*N is computed correctly.
    total_squares = N * N 
    
    # The number of squares where you can place your piece is
    # the total number of squares minus the number of forbidden squares.
    ans = total_squares - num_forbidden_squares
    
    # Print the answer to standard output, followed by a newline.
    sys.stdout.write(str(ans) + "
")

# This is standard Python practice to ensure solve() is called only when the script is executed directly.
if __name__ == '__main__':
    solve()