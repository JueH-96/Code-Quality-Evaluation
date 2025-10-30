n = int(input())
grid_a = [input().strip() for _ in range(n)]
grid_b = [input().strip() for _ in range(n)]

for i in range(n):
    for j in range(n):
        if grid_a[i][j] != grid_b[i][j]:
            print(i + 1, j + 1)
            exit()