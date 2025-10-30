n = int(input())
grid = [list(input().strip()) for _ in range(n)]

for i in range(1, n//2 + 1):
    top = i - 1
    bottom = (n + 1 - i) - 1
    for offset in range(bottom - top):
        temp = grid[top][top + offset]
        grid[top][top + offset] = grid[bottom - offset][top]
        grid[bottom - offset][top] = grid[bottom][bottom - offset]
        grid[bottom][bottom - offset] = grid[top + offset][bottom]
        grid[top + offset][bottom] = temp

for row in grid:
    print(''.join(row))