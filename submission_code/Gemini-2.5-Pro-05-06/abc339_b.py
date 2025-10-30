def solve():
    H, W, N = map(int, input().split())

    # Initialize grid: 0 for white, 1 for black
    # grid[row_idx][col_idx]
    grid = [[0 for _ in range(W)] for _ in range(H)] 

    # Takahashi's initial state
    current_r, current_c = 0, 0  # Starts at (1,1) -> (0,0) in 0-indexed
    
    # Directions: 0:Up, 1:Right, 2:Down, 3:Left
    # dr stores change in row_idx, dc stores change in col_idx
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]
    
    current_dir_idx = 0  # Initially facing Up

    for _ in range(N):
        # Current cell is (current_r, current_c)
        if grid[current_r][current_c] == 0:  # Cell is white
            # 1. Repaint it black
            grid[current_r][current_c] = 1
            # 2. Rotate 90 degrees clockwise
            current_dir_idx = (current_dir_idx + 1) % 4
            # 3. Move forward one cell
            current_r = (current_r + dr[current_dir_idx]) % H
            current_c = (current_c + dc[current_dir_idx]) % W
        else:  # Cell is black (grid[current_r][current_c] == 1)
            # 1. Repaint it white
            grid[current_r][current_c] = 0
            # 2. Rotate 90 degrees counterclockwise
            current_dir_idx = (current_dir_idx - 1 + 4) % 4 # +4 ensures positive before modulo
            # 3. Move forward one cell
            current_r = (current_r + dr[current_dir_idx]) % H
            current_c = (current_c + dc[current_dir_idx]) % W
            
    # Print the final state of the grid
    for r_idx in range(H):
        row_chars = []
        for c_idx in range(W):
            if grid[r_idx][c_idx] == 0:  # White cell
                row_chars.append('.')
            else:  # Black cell
                row_chars.append('#')
        print("".join(row_chars))

# Execute the solution
solve()