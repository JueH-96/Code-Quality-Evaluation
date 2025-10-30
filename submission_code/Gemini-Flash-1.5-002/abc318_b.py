# YOUR CODE HERE
n = int(input())
rects = []
for i in range(n):
    a, b, c, d = map(int, input().split())
    rects.append((a, b, c, d))

grid = [[0 for _ in range(101)] for _ in range(101)]
for a, b, c, d in rects:
    for i in range(a, b):
        for j in range(c, d):
            grid[i][j] = 1

total_area = 0
for i in range(101):
    for j in range(101):
        total_area += grid[i][j]

print(total_area)