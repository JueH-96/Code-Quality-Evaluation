def solve():
    import sys
    data = sys.stdin.read().strip().split()
    H, W, N = map(int, data)

    # 0-based indexing for internally storing rows and columns.
    # We will store colors as: 0 = white, 1 = black
    grid = [[0]*W for _ in range(H)]

    # Directions: up=0, right=1, down=2, left=3
    # We'll store the movement vectors as (d_row, d_col).
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Starting position (0-based) and orientation
    r, c = 0, 0
    dir_idx = 0  # 0 = up
    
    for _ in range(N):
        if grid[r][c] == 0:  # white
            grid[r][c] = 1   # paint black
            dir_idx = (dir_idx + 1) % 4  # turn right
        else:               # black
            grid[r][c] = 0  # paint white
            dir_idx = (dir_idx - 1) % 4  # turn left
        
        # Move forward
        dr, dc = directions[dir_idx]
        r = (r + dr) % H
        c = (c + dc) % W
    
    # Output
    for row in range(H):
        result_row = []
        for col in range(W):
            if grid[row][col] == 0:
                result_row.append('.')
            else:
                result_row.append('#')
        print(''.join(result_row))

# Call solve() to execute
solve()