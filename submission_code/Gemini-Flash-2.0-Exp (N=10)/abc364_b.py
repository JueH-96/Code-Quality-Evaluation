h, w = map(int, input().split())
si, sj = map(int, input().split())
grid = []
for _ in range(h):
    grid.append(list(input()))
x = input()

curr_i = si - 1
curr_j = sj - 1

for move in x:
    if move == 'L':
        if curr_j - 1 >= 0 and grid[curr_i][curr_j - 1] == '.':
            curr_j -= 1
    elif move == 'R':
        if curr_j + 1 < w and grid[curr_i][curr_j + 1] == '.':
            curr_j += 1
    elif move == 'U':
        if curr_i - 1 >= 0 and grid[curr_i - 1][curr_j] == '.':
            curr_i -= 1
    elif move == 'D':
        if curr_i + 1 < h and grid[curr_i + 1][curr_j] == '.':
            curr_i += 1

print(curr_i + 1, curr_j + 1)