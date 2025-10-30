def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    H, W = map(int, data[0].split())
    start_line = data[1].split()
    S_i = int(start_line[0])
    S_j = int(start_line[1])
    grid_lines = data[2:2+H]
    X = data[2+H].strip()
    
    r = S_i - 1
    c = S_j - 1
    
    grid = []
    for i in range(H):
        grid.append(list(grid_lines[i].strip()))
    
    directions = {
        'L': (0, -1),
        'R': (0, 1),
        'U': (-1, 0),
        'D': (1, 0)
    }
    
    for move in X:
        dr, dc = directions[move]
        nr = r + dr
        nc = c + dc
        if 0 <= nr < H and 0 <= nc < W:
            if grid[nr][nc] == '.':
                r, c = nr, nc
    print(f"{r+1} {c+1}")

if __name__ == "__main__":
    main()