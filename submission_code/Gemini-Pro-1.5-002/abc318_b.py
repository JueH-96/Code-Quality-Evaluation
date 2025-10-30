# YOUR CODE HERE
n = int(input())
grid = [[0] * 101 for _ in range(101)]
for _ in range(n):
    a, b, c, d = map(int, input().split())
    for i in range(a, b):
        for j in range(c, d):
            grid[i][j] = 1
area = 0
for i in range(101):
    for j in range(101):
        if grid[i][j] == 1:
            area += 1
print(area)