def main():
    import sys
    data = sys.stdin.read().split()
    # Read H, W
    H = int(data[0])
    W = int(data[1])
    # Read starting position (1-based)
    si = int(data[2])
    sj = int(data[3])
    # Read grid lines
    grid = []
    idx = 4
    for _ in range(H):
        grid.append(data[idx])
        idx += 1
    # Read the move string X
    X = data[idx]
    # Current position
    r, c = si, sj
    # Movement deltas
    moves = {
        'L': (0, -1),
        'R': (0,  1),
        'U': (-1, 0),
        'D': (1,  0)
    }
    # Simulate each move
    for ch in X:
        dr, dc = moves[ch]
        nr, nc = r + dr, c + dc
        # Check bounds and emptiness
        if 1 <= nr <= H and 1 <= nc <= W and grid[nr-1][nc-1] == '.':
            r, c = nr, nc
    # Print final position
    print(r, c)

if __name__ == "__main__":
    main()