n = int(input())
grid = [[False for _ in range(100)] for __ in range(100)]

for _ in range(n):
    a, b, c, d = map(int, input().split())
    for x in range(a, b):
        for y in range(c, d):
            grid[y][x] = True

total = sum(sum(row) for row in grid)
print(total)