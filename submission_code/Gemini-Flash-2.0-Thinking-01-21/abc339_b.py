def solve():
    h, w, n = map(int, input().split())
    grid = [['.' for _ in range(w)] for _ in range(h)]
    current_row = 0
    current_col = 0
    current_direction = 0 # 0: up, 1: right, 2: down, 3: left
    
    for _ in range(n):
        current_color = grid[current_row][current_col]
        if current_color == '.':
            grid[current_row][current_col] = '#'
            current_direction = (current_direction + 1) % 4
        else:
            grid[current_row][current_col] = '.'
            current_direction = (current_direction - 1 + 4) % 4
            
        if current_direction == 0:
            current_row = (current_row - 1 + h) % h
        elif current_direction == 1:
            current_col = (current_col + 1) % w
        elif current_direction == 2:
            current_row = (current_row + 1) % h
        elif current_direction == 3:
            current_col = (current_col - 1 + w) % w
            
    for i in range(h):
        print("".join(grid[i]))

if __name__ == '__main__':
    solve()