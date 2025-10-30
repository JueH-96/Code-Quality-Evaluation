# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H, W = map(int, data[:2])
S_i, S_j = map(int, data[2:4])
S_i -= 1
S_j -= 1
grid = data[4:4+H]
X = data[4+H]

current_pos = [S_i, S_j]

for move in X:
    if move == 'L' and current_pos[1] > 0 and grid[current_pos[0]][current_pos[1] - 1] == '.':
        current_pos[1] -= 1
    elif move == 'R' and current_pos[1] < W - 1 and grid[current_pos[0]][current_pos[1] + 1] == '.':
        current_pos[1] += 1
    elif move == 'U' and current_pos[0] > 0 and grid[current_pos[0] - 1][current_pos[1]] == '.':
        current_pos[0] -= 1
    elif move == 'D' and current_pos[0] < H - 1 and grid[current_pos[0] + 1][current_pos[1]] == '.':
        current_pos[0] += 1

print(current_pos[0] + 1, current_pos[1] + 1)