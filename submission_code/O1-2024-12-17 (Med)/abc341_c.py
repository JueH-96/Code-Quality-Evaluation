def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    H, W, N = map(int, input_data[:3])
    T = input_data[3]
    S = input_data[4:]

    # Build a 2D boolean array indicating whether each cell is land (True) or sea (False)
    land = [[(S[r][c] == '.') for c in range(W)] for r in range(H)]
    
    # Directions map
    moves = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0)
    }
    
    # Start with all land cells as possible initial positions
    # (The perimeter is sea by problem statement, so only interior land cells remain.)
    possible_positions = set()
    for r in range(H):
        for c in range(W):
            if land[r][c]:
                possible_positions.add((r, c))

    # Simulate the moves
    for move in T:
        dr, dc = moves[move]
        next_positions = set()
        for (r, c) in possible_positions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and land[nr][nc]:
                next_positions.add((nr, nc))
        possible_positions = next_positions

    # The answer is how many distinct final positions remain possible
    print(len(possible_positions))

# Do not forget to call main()
if __name__ == "__main__":
    main()