# Read input
h, w = map(int, input().split())
si, sj = map(int, input().split())
grid = [input().strip() for _ in range(h)]
x, y = si, sj
X = input().strip()

# Process each command
for c in X:
    if c == 'U':
        new_x = x - 1
        if new_x >= 1:
            if grid[new_x - 1][y - 1] == '.':
                x = new_x
    elif c == 'D':
        new_x = x + 1
        if new_x <= h:
            if grid[new_x - 1][y - 1] == '.':
                x = new_x
    elif c == 'L':
        new_y = y - 1
        if new_y >= 1:
            if grid[x - 1][new_y - 1] == '.':
                y = new_y
    elif c == 'R':
        new_y = y + 1
        if new_y <= w:
            if grid[x - 1][new_y - 1] == '.':
                y = new_y

# Output the final position
print(x, y)