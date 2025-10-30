import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    occupied_squares = set()
    attacked_squares = set()

    # Knight's moves relative coordinates (dr, dc)
    # These are the 8 possible (delta_row, delta_column) offsets for a knight's move
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    for _ in range(M):
        r, c = map(int, sys.stdin.readline().split())
        occupied_squares.add((r, c)) # Add the current piece's location to occupied squares

        # For each piece, calculate all squares it can attack
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            # Check if the attacked square is within the grid boundaries (1 to N)
            if 1 <= nr <= N and 1 <= nc <= N:
                attacked_squares.add((nr, nc))

    # Calculate total unique unsafe squares using the principle of inclusion-exclusion.
    # The set of unsafe squares is the union of occupied_squares and attacked_squares.
    # |A U B| = |A| + |B| - |A INTERSECT B|

    num_occupied = len(occupied_squares)
    num_attacked = len(attacked_squares)

    # Count squares that are both occupied by a piece AND are also attacked by some piece.
    # This represents the intersection |P INTERSECT A|.
    num_occupied_and_attacked = 0
    for pos in occupied_squares:
        if pos in attacked_squares:
            num_occupied_and_attacked += 1

    # Total number of unique unsafe squares on the grid
    total_unsafe_count = num_occupied + num_attacked - num_occupied_and_attacked

    # Total squares in the N x N grid
    total_grid_squares = N * N

    # The number of safe squares is the total squares minus the unique unsafe squares.
    safe_squares_count = total_grid_squares - total_unsafe_count

    print(safe_squares_count)

# Call the solve function to execute the program logic
solve()