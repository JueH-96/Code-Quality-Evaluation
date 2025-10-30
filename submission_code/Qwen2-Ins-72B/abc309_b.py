n = int(input())
a = [list(map(int, input())) for _ in range(n)]

for i in range(1, n):
    a[0][i], a[i][n-1], a[n-1][n-i], a[n-i][0] = a[i][n-1], a[n-1][n-i], a[n-i][0], a[0][i]

for i in range(n):
    print(''.join(map(str, a[i])))