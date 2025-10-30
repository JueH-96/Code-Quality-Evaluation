def solve():
    N = int(input())
    
    # Create the grid
    grid = [[0] * N for _ in range(N)]
    
    # Place Takahashi at the center
    center = N // 2
    grid[center][center] = 'T'
    
    # Generate spiral path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    
    # Start from top-left corner
    row, col = 0, 0
    dir_idx = 0
    part_num = 1
    
    # Track visited cells
    visited = [[False] * N for _ in range(N)]
    visited[center][center] = True
    
    # Place the dragon parts in a spiral pattern
    while part_num <= N * N - 1:
        # Place current part
        grid[row][col] = part_num
        visited[row][col] = True
        part_num += 1
        
        # Try to continue in current direction
        next_row = row + directions[dir_idx][0]
        next_col = col + directions[dir_idx][1]
        
        # Check if we need to turn
        if (next_row < 0 or next_row >= N or next_col < 0 or next_col >= N or 
            visited[next_row][next_col]):
            # Turn clockwise
            dir_idx = (dir_idx + 1) % 4
            next_row = row + directions[dir_idx][0]
            next_col = col + directions[dir_idx][1]
        
        # Move to next position if valid
        if (0 <= next_row < N and 0 <= next_col < N and 
            not visited[next_row][next_col]):
            row, col = next_row, next_col
        else:
            # Find any unvisited adjacent cell
            found = False
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < N and 0 <= new_col < N and 
                    not visited[new_row][new_col]):
                    row, col = new_row, new_col
                    found = True
                    break
            
            if not found:
                break
    
    # Print the grid
    for row in grid:
        print(' '.join(str(cell) for cell in row))

solve()