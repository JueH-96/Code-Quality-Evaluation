H, W = map(int, input().split())
s_i, s_j = map(int, input().split())
current_row = s_i - 1
current_col = s_j - 1
grid = [input().strip() for _ in range(H)]
X = input().strip()

for move in X:
    new_row, new_col = current_row, current_col
    if move == 'L':
        new_col -= 1
    elif move == 'R':
        new_col += 1
    elif move == 'U':
        new_row -= 1
    elif move == 'D':
        new_row += 1
    if 0 <= new_row < H and 0 <= new_col < W:
        if grid[new_row][new_col] == '.':
            current_row, current_col = new_row, new_col

print(current_row + 1, current_col + 1)