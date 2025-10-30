def arrange_dragon_parts(N):
    # Create an empty grid
    grid = [['' for _ in range(N)] for _ in range(N)]
    
    # Place Takahashi in the center
    center = (N // 2, N // 2)
    grid[center[0]][center[1]] = 'T'
    
    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Start placing dragon parts from 1 to N^2 - 1
    current_part = 1
    x, y = center
    
    # Use a queue to perform a breadth-first search for placement
    from collections import deque
    queue = deque([(x, y)])
    
    while queue and current_part < N * N:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < N and 0 <= ny < N and grid[nx][ny] == '':
                grid[nx][ny] = current_part
                current_part += 1
                queue.append((nx, ny))
    
    # Print the grid
    for row in grid:
        print(' '.join(str(cell) for cell in row))

# Read input
N = int(input().strip())
arrange_dragon_parts(N)