n = int(input())
grid = [[False] * 100 for _ in range(100)]

for _ in range(n):
    a, b, c, d = map(int, input().split())
    for x in range(a, b):
        for y in range(c, d):
            grid[x][y] = True

count = 0
for row in grid:
    count += sum(row)

print(count)