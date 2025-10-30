H, W = map(int, input().split())
si, sj = map(int, input().split())
current_i = si - 1
current_j = sj - 1
grid = [list(input().strip()) for _ in range(H)]
X = input().strip()

for c in X:
    if c == 'L':
        new_j = current_j - 1
        if new_j >= 0 and grid[current_i][new_j] == '.':
            current_j = new_j
    elif c == 'R':
        new_j = current_j + 1
        if new_j < W and grid[current_i][new_j] == '.':
            current_j = new_j
    elif c == 'U':
        new_i = current_i - 1
        if new_i >= 0 and grid[new_i][current_j] == '.':
            current_i = new_i
    elif c == 'D':
        new_i = current_i + 1
        if new_i < H and grid[new_i][current_j] == '.':
            current_i = new_i

print(current_i + 1, current_j + 1)