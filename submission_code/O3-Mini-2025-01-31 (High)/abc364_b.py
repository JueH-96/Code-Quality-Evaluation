def main():
    import sys
    input = sys.stdin.readline
    
    # read grid dimensions
    H, W = map(int, input().strip().split())
    # read starting coordinates (1-indexed)
    si, sj = map(int, input().strip().split())
    # convert to 0-indexed for internal processing
    si -= 1
    sj -= 1
    
    # read the grid
    grid = [input().strip() for _ in range(H)]
    
    # read the moves string
    moves = input().strip()
    
    # current position (0-indexed)
    r, c = si, sj
    
    for move in moves:
        nr, nc = r, c
        if move == 'L':
            nc = c - 1
        elif move == 'R':
            nc = c + 1
        elif move == 'U':
            nr = r - 1
        elif move == 'D':
            nr = r + 1
        
        # check boundaries and whether cell is empty
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.':
            r, c = nr, nc
    
    # print final position in 1-indexed format
    print(r + 1, c + 1)

if __name__ == '__main__':
    main()