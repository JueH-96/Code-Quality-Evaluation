import sys

def main() -> None:
    read = sys.stdin.readline

    # Read basic parameters
    H, W, X, Y = map(int, read().split())

    # Read the grid (1-indexed for convenience)
    grid = [None] * (H + 1)              # grid[1] .. grid[H] are valid
    for i in range(1, H + 1):
        line = read().rstrip('
')
        grid[i] = [' '] + list(line)     # grid[i][1] .. grid[i][W]

    # Movement instruction string
    T = read().strip()

    # Direction vectors
    dx = {'U': -1, 'D':  1, 'L': 0, 'R': 0}
    dy = {'U':  0, 'D':  0, 'L':-1, 'R': 1}

    # Simulation
    x, y = X, Y
    visited_houses = set()               # set of (row, col) pairs

    # (Starting cell is guaranteed not to be a house, but check anyway)
    if grid[x][y] == '@':
        visited_houses.add((x, y))

    for ch in T:
        nx, ny = x + dx[ch], y + dy[ch]

        # Move only if the target cell is passable
        if grid[nx][ny] != '#':
            x, y = nx, ny
            if grid[x][y] == '@':
                visited_houses.add((x, y))

    # Output final position and number of distinct houses visited
    print(x, y, len(visited_houses))


if __name__ == "__main__":
    main()