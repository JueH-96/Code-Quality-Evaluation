import sys

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
S_i = int(data[2])
S_j = int(data[3])

grid = []
index = 4
for i in range(H):
    grid.append(data[index])
    index += 1

X = data[index]

current_i = S_i - 1
current_j = S_j - 1

for move in X:
    if move == 'L' and current_j > 0 and grid[current_i][current_j - 1] == '.':
        current_j -= 1
    elif move == 'R' and current_j < W - 1 and grid[current_i][current_j + 1] == '.':
        current_j += 1
    elif move == 'U' and current_i > 0 and grid[current_i - 1][current_j] == '.':
        current_i -= 1
    elif move == 'D' and current_i < H - 1 and grid[current_i + 1][current_j] == '.':
        current_i += 1

print(current_i + 1, current_j + 1)