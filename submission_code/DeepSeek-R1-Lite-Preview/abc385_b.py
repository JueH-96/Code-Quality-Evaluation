def main():
    import sys

    # Read H, W, X, Y
    H_W_X_Y = sys.stdin.readline().split()
    H = int(H_W_X_Y[0])
    W = int(H_W_X_Y[1])
    X = int(H_W_X_Y[2]) - 1  # Convert to 0-based indexing
    Y = int(H_W_X_Y[3]) - 1

    # Read the grid
    grid = []
    for _ in range(H):
        row = sys.stdin.readline().strip()
        grid.append(row)

    # Read the movement string T
    T = sys.stdin.readline().strip()

    # Initialize set to track unique houses visited
    houses = set()

    # Directions mapping
    move = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

    current_x = X
    current_y = Y

    # Simulate the movement
    for direction in T:
        dx, dy = move.get(direction, (0, 0))
        new_x = current_x + dx
        new_y = current_y + dy

        # Check if new position is within bounds and not a wall
        if 0 <= new_x < H and 0 <= new_y < W and grid[new_x][new_y] != '#':
            current_x = new_x
            current_y = new_y

        # Check if current cell contains a house
        if grid[current_x][current_y] == '@':
            houses.add((current_x, current_y))

    # Convert back to 1-based indexing for output
    final_X = current_x + 1
    final_Y = current_y + 1
    C = len(houses)

    print(final_X, final_Y, C)

if __name__ == "__main__":
    main()