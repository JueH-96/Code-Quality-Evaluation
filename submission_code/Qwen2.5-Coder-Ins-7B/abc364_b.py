# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
S_i = int(data[2]) - 1
S_j = int(data[3]) - 1
grid = [list(data[i + 4]) for i in range(H)]
X = data[-1]

x, y = S_i, S_j
for move in X:
    if move == 'L' and y > 0 and grid[x][y - 1] == '.':
        y -= 1
    elif move == 'R' and y < W - 1 and grid[x][y + 1] == '.':
        y += 1
    elif move == 'U' and x > 0 and grid[x - 1][y] == '.':
        x -= 1
    elif move == 'D' and x < H - 1 and grid[x + 1][y] == '.':
        x += 1

print(x + 1, y + 1)