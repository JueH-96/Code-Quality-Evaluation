# YOUR CODE HERE
H, W, X, Y = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]
T = input().strip()

# Adjust to 0-based index
x, y = X-1, Y-1

# Initialize visited houses
visited_houses = set()
if grid[x][y] == '@':
    visited_houses.add((x, y))

for move in T:
    if move == 'U':
        nx, ny = x-1, y
    elif move == 'D':
        nx, ny = x+1, y
    elif move == 'L':
        nx, ny = x, y-1
    elif move == 'R':
        nx, ny = x, y+1
    else:
        continue
    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
        x, y = nx, ny
        if grid[x][y] == '@':
            visited_houses.add((x, y))

print(x+1, y+1, len(visited_houses))