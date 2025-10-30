def solve():
    n = int(input())
    a = [[[0] * n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row = list(map(int, input().split()))
            for k in range(n):
                a[i][j][k] = row[k]

    s = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            for z in range(1, n + 1):
                s[x][y][z] = a[x - 1][y - 1][z - 1] + \
                             s[x - 1][y][z] + s[x][y - 1][z] + s[x][y][z - 1] - \
                             s[x - 1][y - 1][z] - s[x - 1][y][z - 1] - s[x][y - 1][z - 1] + \
                             s[x - 1][y - 1][z - 1]

    q = int(input())
    for _ in range(q):
        lx, rx, ly, ry, lz, rz = map(int, input().split())
        ans = s[rx][ry][rz] - s[lx - 1][ry][rz] - s[rx][ly - 1][rz] - s[rx][ry][lz - 1] + \
              s[lx - 1][ly - 1][rz] + s[lx - 1][ry][lz - 1] + s[rx][ly - 1][lz - 1] - \
              s[lx - 1][ly - 1][lz - 1]
        print(ans)

if __name__ == "__main__":
    solve()