def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = [data[i + 2] for i in range(H)]
    
    # Find min_row
    min_row = -1
    for i in range(H):
        if '#' in grid[i]:
            min_row = i
            break
    
    # Find max_row
    max_row = -1
    for i in range(H-1, -1, -1):
        if '#' in grid[i]:
            max_row = i
            break
    
    # Find min_col
    min_col = -1
    for j in range(W):
        for i in range(H):
            if grid[i][j] == '#':
                min_col = j
                break
        if min_col != -1:
            break
    
    # Find max_col
    max_col = -1
    for j in range(W-1, -1, -1):
        for i in range(H):
            if grid[i][j] == '#':
                max_col = j
                break
        if max_col != -1:
            break
    
    # Iterate through the rectangle to find the '.' cell
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                # 1-based indexing
                print(i + 1, j + 1)
                return

if __name__ == "__main__":
    main()