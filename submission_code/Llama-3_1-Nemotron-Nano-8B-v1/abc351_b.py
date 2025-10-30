n = int(input())
a = [input() for _ in range(n)]
b = [input() for _ in range(n)]
for i in range(n):
    for j in range(n):
        if a[i][j] != b[i][j]:
            print(i+1, j+1)
            exit()