def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    # Directions: 0 = up, 1 = right, 2 = down, 3 = left
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    # Initialize the grid
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    # Start position and direction
    x, y = 0, 0
    direction = 0
    
    for _ in range(N):
        if grid[y][x] == '.':
            # Paint black, turn clockwise
            grid[y][x] = '#'
            direction = (direction + 1) % 4
        else:
            # Paint white, turn counterclockwise
            grid[y][x] = '.'
            direction = (direction - 1) % 4
        
        # Move forward
        dx, dy = directions[direction]
        x = (x + dx) % W
        y = (y + dy) % H
    
    # Output the grid
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()