def solve():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Create an empty grid
    grid = [[''] * N for _ in range(N)]
    
    # Center position for Takahashi
    center = (N // 2, N // 2)
    grid[center[0]][center[1]] = 'T'
    
    # Directions for spiral: right, down, left, up
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Start from the center
    x, y = center
    num = 1
    direction_index = 0
    
    # We need to place N^2 - 1 parts
    for _ in range(N**2 - 1):
        # Move in the current direction
        dx, dy = directions[direction_index]
        x += dx
        y += dy
        
        # If out of bounds or cell is already filled, adjust the direction
        while not (0 <= x < N and 0 <= y < N and grid[x][y] == ''):
            # Move back
            x -= dx
            y -= dy
            # Change direction
            direction_index = (direction_index + 1) % 4
            dx, dy = directions[direction_index]
            # Move in the new direction
            x += dx
            y += dy
        
        # Place the part number
        grid[x][y] = str(num)
        num += 1
    
    # Output the grid
    for row in grid:
        print(" ".join(row))