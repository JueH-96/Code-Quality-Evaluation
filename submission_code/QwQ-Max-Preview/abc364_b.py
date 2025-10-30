h, w = map(int, input().split())
current_x, current_y = map(int, input().split())
grid = [input().strip() for _ in range(h)]
x_str = input().strip()

for c in x_str:
    new_x, new_y = current_x, current_y
    if c == 'L':
        new_y -= 1
    elif c == 'R':
        new_y += 1
    elif c == 'U':
        new_x -= 1
    elif c == 'D':
        new_x += 1
    # Check if new position is valid
    if 1 <= new_x <= h and 1 <= new_y <= w:
        if grid[new_x - 1][new_y - 1] == '.':
            current_x, current_y = new_x, new_y

print(current_x, current_y)