# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    N = int(input[2])
    
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    # Directions: 0 = up, 1 = right, 2 = down, 3 = left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    x, y = 0, 0
    direction = 0
    
    for _ in range(N):
        if grid[x][y] == '.':
            grid[x][y] = '#'
            direction = (direction + 1) % 4
        else:
            grid[x][y] = '.'
            direction = (direction - 1) % 4
        
        dx, dy = directions[direction]
        x = (x + dx) % H
        y = (y + dy) % W
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()