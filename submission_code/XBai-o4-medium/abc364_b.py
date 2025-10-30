h, w = map(int, input().split())
si, sj = map(int, input().split())
grid = [input().strip() for _ in range(h)]
x = input().strip()

current_row = si - 1
current_col = sj - 1

for c in x:
    dr = 0
    dc = 0
    if c == 'U':
        dr = -1
    elif c == 'D':
        dr = 1
    elif c == 'L':
        dc = -1
    elif c == 'R':
        dc = 1
    
    new_row = current_row + dr
    new_col = current_col + dc
    
    if 0 <= new_row < h and 0 <= new_col < w:
        if grid[new_row][new_col] == '.':
            current_row = new_row
            current_col = new_col

print(current_row + 1, current_col + 1)