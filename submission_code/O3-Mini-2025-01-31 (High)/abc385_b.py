def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Parse the first 4 tokens: H, W, X, Y
    H = int(data[0])
    W = int(data[1])
    X = int(data[2])
    Y = int(data[3])
    idx = 4

    # Read grid rows (each row is a string)
    grid = []
    for i in range(H):
        grid.append(data[idx])
        idx += 1

    # The move string T
    T = data[idx] if idx < len(data) else ""

    # Convert initial position to 0-indexing.
    x, y = X - 1, Y - 1

    # Set to store distinct houses passed (cells where grid cell=='@')
    visited_houses = set()

    # Movement dictionary for each direction.
    moves = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }

    # Process each move in T.
    for move in T:
        dx, dy = moves[move]
        nx, ny = x + dx, y + dy
        # Check boundaries and whether the target cell is passable.
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
            x, y = nx, ny
        # If the current cell contains a house, record it.
        if grid[x][y] == '@':
            visited_houses.add((x, y))
    
    # Convert final position back to 1-indexed coordinates.
    final_x, final_y = x + 1, y + 1
    count = len(visited_houses)

    sys.stdout.write(f"{final_x} {final_y} {count}")

if __name__ == '__main__':
    main()