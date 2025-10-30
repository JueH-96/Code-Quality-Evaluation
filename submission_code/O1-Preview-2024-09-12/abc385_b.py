# YOUR CODE HERE
H, W, X, Y = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())
T = input()

x, y = X - 1, Y - 1  # converting to 0-based indexing

dx = {'U': -1, 'D': 1, 'L': 0, 'R': 0}
dy = {'U': 0, 'D': 0, 'L': -1, 'R': 1}

houses_passed = set()

for move in T:
    nx = x + dx[move]
    ny = y + dy[move]
    # Check if next position is within bounds and passable
    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
        x, y = nx, ny  # move to next position
        if grid[x][y] == '@':
            houses_passed.add((x, y))
    else:
        # Stay in the same position
        if grid[x][y] == '@':
            houses_passed.add((x, y))

# After all moves, check if the final position is a house
# Even though already added in the loop, but to be safe
if grid[x][y] == '@':
    houses_passed.add((x, y))

print(x + 1, y + 1, len(houses_passed))