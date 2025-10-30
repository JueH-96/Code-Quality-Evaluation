directions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

H, W = map(int, input().split())
Si, Sj = map(int, input().split())
grid = [input().strip() for _ in range(H)]
X = input().strip()

current_i, current_j = Si, Sj

for cmd in X:
    di, dj = directions[cmd]
    ni = current_i + di
    nj = current_j + dj
    if 1 <= ni <= H and 1 <= nj <= W:
        if grid[ni-1][nj-1] == '.':
            current_i, current_j = ni, nj

print(current_i, current_j)