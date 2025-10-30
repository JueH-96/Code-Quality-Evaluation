# Read input
H, W = map(int, input().split())
Si, Sj = map(int, input().split())
grid = [list(input()) for _ in range(H)]
X = input()

# Simulate Takahashi's movements
x, y = Si, Sj
for action in X:
    if action == 'L' and y > 1 and grid[x-1][y-1] == '.':
        y -= 1
    elif action == 'R' and y < W and grid[x-1][y] == '.':
        y += 1
    elif action == 'U' and x > 1 and grid[x-2][y-1] == '.':
        x -= 1
    elif action == 'D' and x < H and grid[x][y-1] == '.':
        x += 1

# Print the final position
print(x, y)