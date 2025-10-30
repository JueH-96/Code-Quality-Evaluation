H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
grid = []
for _ in range(H):
    line = input().strip()
    grid.append(line)
X = input().strip()

current_x = S_i
current_y = S_j

for c in X:
    if c == 'L':
        if current_y > 1:
            target_y = current_y - 1
            if grid[current_x - 1][target_y - 1] == '.':
                current_y = target_y
    elif c == 'R':
        if current_y < W:
            target_y = current_y + 1
            if grid[current_x - 1][target_y - 1] == '.':
                current_y = target_y
    elif c == 'U':
        if current_x > 1:
            target_x = current_x - 1
            if grid[target_x - 1][current_y - 1] == '.':
                current_x = target_x
    elif c == 'D':
        if current_x < H:
            target_x = current_x + 1
            if grid[target_x - 1][current_y - 1] == '.':
                current_x = target_x

print(current_x, current_y)