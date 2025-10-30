def count_possible_positions(H, W, N, T, S):
    """
    Count the number of possible positions of Takahashi's spaceship after N moves.

    Args:
    H (int): The number of rows in the grid.
    W (int): The number of columns in the grid.
    N (int): The number of moves.
    T (str): The string of moves (L, R, U, D).
    S (list[str]): The grid of land and sea cells.

    Returns:
    int: The number of possible positions.
    """
    # Initialize a set to store the possible positions
    possible_positions = set()

    # Iterate over all cells in the grid
    for i in range(H):
        for j in range(W):
            # Check if the cell is land
            if S[i][j] == '.':
                # Initialize the current position
                x, y = i, j

                # Simulate the moves
                for move in T:
                    # Move left
                    if move == 'L':
                        y -= 1
                    # Move right
                    elif move == 'R':
                        y += 1
                    # Move up
                    elif move == 'U':
                        x -= 1
                    # Move down
                    elif move == 'D':
                        x += 1

                    # Check if the new position is within the grid and is land
                    if 0 <= x < H and 0 <= y < W and S[x][y] == '.':
                        # Update the current position
                        continue
                    else:
                        # If the new position is not valid, break the loop
                        break
                else:
                    # If all moves are valid, add the final position to the set
                    possible_positions.add((x, y))

    # Return the number of possible positions
    return len(possible_positions)


# Read the input
H, W, N = map(int, input().split())
T = input()
S = [input() for _ in range(H)]

# Count the possible positions
possible_positions = count_possible_positions(H, W, N, T, S)

# Print the result
print(possible_positions)