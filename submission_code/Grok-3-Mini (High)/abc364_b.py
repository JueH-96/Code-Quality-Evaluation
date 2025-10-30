import sys
data = sys.stdin.read().split()
H = int(data[0])
W = int(data[1])
start_i = int(data[2])
start_j = int(data[3])
grid = data[4:4+H]
X = data[4+H]
current_row = start_i - 1
current_col = start_j - 1
for move in X:
    if move == 'L':
        if current_col > 0 and grid[current_row][current_col - 1] == '.':
            current_col -= 1
    elif move == 'R':
        if current_col < W - 1 and grid[current_row][current_col + 1] == '.':
            current_col += 1
    elif move == 'U':
        if current_row > 0 and grid[current_row - 1][current_col] == '.':
            current_row -= 1
    elif move == 'D':
        if current_row < H - 1 and grid[current_row + 1][current_col] == '.':
            current_row += 1
print(current_row + 1, current_col + 1)