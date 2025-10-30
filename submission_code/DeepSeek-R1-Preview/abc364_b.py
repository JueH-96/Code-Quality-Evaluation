h, w = map(int, input().split())
si, sj = map(int, input().split())
current_i = si - 1
current_j = sj - 1

grid = [input().strip() for _ in range(h)]
x = input().strip()

direction_map = {
    'L': (0, -1),
    'R': (0, 1),
    'U': (-1, 0),
    'D': (1, 0)
}

for c in x:
    di, dj = direction_map[c]
    new_i = current_i + di
    new_j = current_j + dj
    if 0 <= new_i < h and 0 <= new_j < w:
        if grid[new_i][new_j] == '.':
            current_i, current_j = new_i, new_j

print(current_i + 1, current_j + 1)