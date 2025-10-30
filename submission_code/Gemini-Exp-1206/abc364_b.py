H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())
X = input()

curr_i, curr_j = S_i - 1, S_j - 1

for move in X:
    if move == 'L':
        if curr_j > 0 and grid[curr_i][curr_j - 1] == '.':
            curr_j -= 1
    elif move == 'R':
        if curr_j < W - 1 and grid[curr_i][curr_j + 1] == '.':
            curr_j += 1
    elif move == 'U':
        if curr_i > 0 and grid[curr_i - 1][curr_j] == '.':
            curr_i -= 1
    elif move == 'D':
        if curr_i < H - 1 and grid[curr_i + 1][curr_j] == '.':
            curr_i += 1

print(curr_i + 1, curr_j + 1)