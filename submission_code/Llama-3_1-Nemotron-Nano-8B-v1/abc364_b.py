H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
x = S_i - 1
y = S_j - 1
grid = [input().strip() for _ in range(H)]
X = input().strip()

for move in X:
    if move == 'L':
        if y > 0 and grid[x][y-1] == '.':
            y -= 1
    elif move == 'R':
        if y < W - 1 and grid[x][y+1] == '.':
            y += 1
    elif move == 'U':
        if x > 0 and grid[x-1][y] == '.':
            x -= 1
    elif move == 'D':
        if x < H - 1 and grid[x+1][y] == '.':
            x += 1

print(x + 1, y + 1)