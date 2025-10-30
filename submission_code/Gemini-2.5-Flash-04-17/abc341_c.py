import sys

def solve():
    """
    Reads the grid dimensions, move sequence, and grid layout.
    Calculates the number of unique cells that could be the final position
    after following the move sequence from any valid starting land cell.
    """
    # Read H, W, N (Height, Width, Number of moves)
    H, W, N = map(int, sys.stdin.readline().split())

    # Read the move sequence T
    T = sys.stdin.readline().strip()

    # Read the grid S. S is a list of strings, where S[i] is the (i+1)-th row (0-based index)
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Use a set to store unique possible final positions (as tuples (row, col))
    possible_final_positions = set()

    # Iterate through every cell in the grid as a potential starting cell
    # Grid coordinates are 0-based: row from 0 to H-1, column from 0 to W-1
    for start_r in range(H):
        for start_c in range(W):
            # A cell can be a landing spot only if it is land ('.')
            if S[start_r][start_c] == '#':
                continue # Cannot start on sea

            # If the starting cell is land, simulate the path from this cell
            current_r, current_c = start_r, start_c
            is_valid_path = True # Assume the path is valid initially

            # Simulate each move in the sequence T
            for move in T:
                next_r, next_c = current_r, current_c

                # Determine the next cell based on the move instruction
                if move == 'U':
                    next_r -= 1
                elif move == 'D':
                    next_r += 1
                elif move == 'L':
                    next_c -= 1
                elif move == 'R':
                    next_c += 1

                # Check if the calculated next cell is valid
                # A cell is valid if it is within grid bounds AND it is land ('.')
                # Must be within grid bounds: 0 <= next_r < H and 0 <= next_c < W
                # Must be land: S[next_r][next_c] == '.'
                # The problem guarantees perimeter cells are sea, so if the calculated
                # next cell is on the perimeter, S[next_r][next_c] will be '#'.
                # So, checking S[next_r][next_c] == '.' AFTER checking bounds
                # ensures the cell is both in bounds and land.
                if 0 <= next_r < H and 0 <= next_c < W and S[next_r][next_c] == '.':
                    # The move is valid. Update the current position.
                    current_r, current_c = next_r, next_c
                else:
                    # The move is invalid (hit sea or went out of bounds).
                    # This starting cell does not result in a valid path.
                    is_valid_path = False
                    break # Stop simulating for this starting cell

            # After simulating all moves (or breaking early due to invalid move)
            if is_valid_path:
                # If the path was valid for all N moves, the final position is a possible current position
                # Add the final position (as a tuple) to the set
                possible_final_positions.add((current_r, current_c))

    # The answer is the number of unique final positions collected in the set
    print(len(possible_final_positions))

# Call the solve function to run the program
solve()