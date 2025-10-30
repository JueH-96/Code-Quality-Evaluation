import sys

def main() -> None:
    # Read grid dimensions
    H_W = sys.stdin.readline().strip()
    while H_W == '':
        H_W = sys.stdin.readline().strip()
    H, W = map(int, H_W.split())

    # Read starting position (1-indexed in statement)
    S_i, S_j = map(int, sys.stdin.readline().split())

    # Read the grid
    grid = [sys.stdin.readline().rstrip('
') for _ in range(H)]

    # Read the move string
    X = sys.stdin.readline().strip()

    # Convert starting position to 0-indexed
    r, c = S_i - 1, S_j - 1

    # Directions: (dr, dc) for each possible command
    moves = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0)
    }

    for ch in X:
        dr, dc = moves[ch]
        nr, nc = r + dr, c + dc
        # Check bounds and emptiness
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.':
            r, c = nr, nc  # Move

    # Output result in 1-indexed coordinates
    print(r + 1, c + 1)

if __name__ == "__main__":
    main()