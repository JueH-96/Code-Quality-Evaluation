# YOUR CODE HERE
n = int(input())
a = [[[0] * n for _ in range(n)] for _ in range(n)]
for i in range(n):
    for j in range(n):
        row = list(map(int, input().split()))
        for k in range(n):
            a[i][j][k] = row[k]

q = int(input())
for _ in range(q):
    lx, rx, ly, ry, lz, rz = map(int, input().split())
    ans = 0
    for i in range(lx - 1, rx):
        for j in range(ly - 1, ry):
            for k in range(lz - 1, rz):
                ans += a[i][j][k]
    print(ans)