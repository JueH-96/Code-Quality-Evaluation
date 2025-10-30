def solve():
    h, w, n = map(int, input().split())
    grid = [['.' for _ in range(w)] for _ in range(h)]
    
    row, col = 0, 0
    direction = 0  # 0: up, 1: right, 2: down, 3: left
    
    for _ in range(n):
        if grid[row][col] == '.':
            grid[row][col] = '#'
            direction = (direction + 1) % 4
        else:
            grid[row][col] = '.'
            direction = (direction - 1 + 4) % 4
        
        if direction == 0:
            row = (row - 1 + h) % h
        elif direction == 1:
            col = (col + 1) % w
        elif direction == 2:
            row = (row + 1) % h
        elif direction == 3:
            col = (col - 1 + w) % w
    
    for row in grid:
        print("".join(row))

solve()