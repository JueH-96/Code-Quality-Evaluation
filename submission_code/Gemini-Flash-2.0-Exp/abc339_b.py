def solve():
    H, W, N = map(int, input().split())
    grid = [['.' for _ in range(W)] for _ in range(H)]
    
    row, col = 0, 0
    direction = 0  # 0: up, 1: right, 2: down, 3: left
    
    for _ in range(N):
        if grid[row][col] == '.':
            grid[row][col] = '#'
            direction = (direction + 1) % 4
        else:
            grid[row][col] = '.'
            direction = (direction - 1) % 4
        
        if direction == 0:
            row = (row - 1) % H
        elif direction == 1:
            col = (col + 1) % W
        elif direction == 2:
            row = (row + 1) % H
        else:
            col = (col - 1) % W
    
    for i in range(H):
        print("".join(grid[i]))

solve()