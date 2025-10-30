def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    
    # Find the bounds of the initial black cells
    min_row, max_row = H, -1
    min_col, max_col = W, -1
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < min_row:
                    min_row = i
                if i > max_row:
                    max_row = i
                if j < min_col:
                    min_col = j
                if j > max_col:
                    max_col = j
    
    # Check if all cells within the rectangle are either '#' or '?'
    for i in range(min_row, max_row + 1):
        for j in range(min_col, max_col + 1):
            if grid[i][j] == '.':
                print("No")
                return
    
    # Check if all '#' outside the rectangle are actually isolated by '.'
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if not (min_row <= i <= max_row and min_col <= j <= max_col):
                    print("No")
                    return
    
    print("Yes")