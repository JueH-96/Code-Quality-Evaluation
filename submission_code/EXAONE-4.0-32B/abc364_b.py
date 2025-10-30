def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    H, W = map(int, data[0].split())
    start_line = data[1].split()
    start_i = int(start_line[0])
    start_j = int(start_line[1])
    
    grid_lines = []
    for i in range(2, 2 + H):
        grid_lines.append(data[i].strip())
    
    X = data[2 + H].strip()
    
    current_i = start_i - 1
    current_j = start_j - 1
    
    for move in X:
        ni, nj = current_i, current_j
        if move == 'L':
            nj -= 1
        elif move == 'R':
            nj += 1
        elif move == 'U':
            ni -= 1
        elif move == 'D':
            ni += 1
            
        if 0 <= ni < H and 0 <= nj < W:
            if grid_lines[ni][nj] == '.':
                current_i, current_j = ni, nj
    
    print(f"{current_i + 1} {current_j + 1}")

if __name__ == "__main__":
    main()