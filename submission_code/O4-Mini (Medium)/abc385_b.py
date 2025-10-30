import sys

def main():
    input = sys.stdin.readline
    H, W, X, Y = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]
    T = input().strip()

    # Current position (1-based). Convert to 0-based indices for grid access.
    x, y = X - 1, Y - 1

    # Set to track distinct houses visited
    visited_houses = set()

    # Movement deltas
    moves = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }

    for c in T:
        dx, dy = moves[c]
        nx, ny = x + dx, y + dy
        # Check passability
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
            x, y = nx, ny
            if grid[x][y] == '@':
                visited_houses.add((x, y))

    # Convert back to 1-based coordinates
    final_X = x + 1
    final_Y = y + 1
    count = len(visited_houses)
    print(final_X, final_Y, count)

if __name__ == "__main__":
    main()