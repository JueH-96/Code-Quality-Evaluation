import sys

def solve():
    # Read H and W (height and width of the grid)
    H, W = map(int, sys.stdin.readline().split())

    # Read the grid content into a list of strings
    # S[i][j] will give the character at 0-indexed row i, column j
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # The target word we are looking for
    target_word = "snuke"

    # Define all 8 possible directions for movement (dr, dc)
    # dr: change in row, dc: change in column
    # These cover horizontal, vertical, and diagonal movements.
    # The (0,0) direction is implicitly excluded because it means no movement,
    # and cells in the sequence must be distinct.
    directions = []
    for dr_val in [-1, 0, 1]:
        for dc_val in [-1, 0, 1]:
            if dr_val == 0 and dc_val == 0:
                continue # Skip the no-movement case
            directions.append((dr_val, dc_val))

    # Iterate through every cell in the grid to find a potential starting point (A_1)
    # r_start, c_start are 0-indexed coordinates
    for r_start in range(H):
        for c_start in range(W):
            # Check if the character at the current cell matches the first letter of "snuke" ('s')
            if S[r_start][c_start] != target_word[0]:
                continue # If not 's', this cell cannot be the start of "snuke"

            # If it is 's', try to form the word "snuke" in all 8 possible directions
            for dr, dc in directions:
                path_found = True
                current_cells_coords = [] # List to store the 1-indexed (row, col) of the found path

                # Check the next 4 cells along this direction to complete the "snuke" sequence
                # k = 0 corresponds to 's', k = 1 to 'n', ..., k = 4 to 'e'
                for k in range(5):
                    curr_r = r_start + k * dr
                    curr_c = c_start + k * dc

                    # Condition 1: Check if the current cell (curr_r, curr_c) is within grid boundaries
                    if not (0 <= curr_r < H and 0 <= curr_c < W):
                        path_found = False # The path goes out of bounds
                        break

                    # Condition 2: Check if the character at the current cell matches the expected character in "snuke"
                    if S[curr_r][curr_c] != target_word[k]:
                        path_found = False # Character mismatch
                        break

                    # If both conditions pass, add the cell to the current path.
                    # Output requires 1-indexed coordinates, so add 1 to row and column.
                    current_cells_coords.append((curr_r + 1, curr_c + 1))

                # If path_found is still True after checking all 5 characters, we have found the unique sequence
                if path_found:
                    # Print the coordinates of the 5 cells, each on a new line
                    for cell_r, cell_c in current_cells_coords:
                        print(cell_r, cell_c)
                    # The problem guarantees a unique solution, so we can exit immediately after finding it
                    return

# Call the main solve function to run the program
solve()