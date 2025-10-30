# YOUR CODE HERE
def main():
    import sys

    H, W = map(int, sys.stdin.readline().split())
    S_i, S_j = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    X = sys.stdin.read().strip()

    # Directions: U, D, L, R
    dir_map = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1),
    }

    # Convert to 0-based indices
    x, y = S_i -1, S_j -1

    for move in X:
        dx, dy = dir_map.get(move, (0,0))
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
            x, y = nx, ny

    # Convert back to 1-based indices
    print(x +1, y +1)

if __name__ == "__main__":
    main()