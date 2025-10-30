H, W = map(int, input().split())
grid = [list(input()) for _ in range(H)]

top = left = float('inf')
bottom = right = -float('inf')

for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            top = min(top, i)
            bottom = max(bottom, i)
            left = min(left, j)
            right = max(right, j)

for i in range(top, bottom+1):
    for j in range(left, right+1):
        if grid[i][j] == '.':
            print(i+1, j+1)
            break