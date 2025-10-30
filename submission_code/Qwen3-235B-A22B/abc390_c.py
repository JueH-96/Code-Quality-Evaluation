def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    grid = []
    for _ in range(H):
        grid.append(input[idx])
        idx += 1
    
    hash_positions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                hash_positions.append((i, j))
    
    if not hash_positions:
        print("No")
        return
    
    min_row = min(i for i, j in hash_positions)
    max_row = max(i for i, j in hash_positions)
    min_col = min(j for i, j in hash_positions)
    max_col = max(j for i, j in hash_positions)
    
    valid = True
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                valid = False
                break
        if not valid:
            break
    
    print("Yes" if valid else "No")

if __name__ == "__main__":
    main()