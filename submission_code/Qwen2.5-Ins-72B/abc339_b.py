# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    # Initialize the grid
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    direction = 0  # Start facing up
    x, y = 0, 0  # Start at (1, 1) which is (0, 0) in 0-indexed
    
    for _ in range(N):
        if grid[x][y] == '.':
            grid[x][y] = '#'
            direction = (direction + 1) % 4  # Rotate 90 degrees clockwise
        else:
            grid[x][y] = '.'
            direction = (direction - 1) % 4  # Rotate 90 degrees counterclockwise
        
        dx, dy = directions[direction]
        x = (x + dx) % H
        y = (y + dy) % W
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()