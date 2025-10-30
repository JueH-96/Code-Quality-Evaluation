def main():
    import sys
    input = sys.stdin.readline
    
    # Read grid dimensions
    H, W = map(int, input().split())
    
    # Read starting position (convert to 0-indexed)
    S_i, S_j = map(int, input().split())
    cur_i, cur_j = S_i - 1, S_j - 1
    
    # Read grid
    grid = [list(input().strip()) for _ in range(H)]
    
    # Read moves
    X = input().strip()
    
    # Define movements
    moves = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    # Process each move
    for move in X:
        di, dj = moves[move]
        ni, nj = cur_i + di, cur_j + dj
        # Check boundary and if the cell is empty.
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
            cur_i, cur_j = ni, nj
    
    # Convert result to 1-indexed
    print(cur_i + 1, cur_j + 1)

if __name__ == '__main__':
    main()