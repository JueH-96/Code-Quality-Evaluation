import sys

def main():
    n = int(sys.stdin.readline())
    a = [[[0]*(n+2) for _ in range(n+2)] for __ in range(n+2)]
    for x in range(1, n+1):
        for y in range(1, n+1):
            row = list(map(int, sys.stdin.readline().split()))
            for z in range(1, n+1):
                a[x][y][z] = row[z-1]
    
    pre = [[[0]*(n+2) for _ in range(n+2)] for __ in range(n+2)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            for k in range(1, n+1):
                pre[i][j][k] = a[i][j][k]
                pre[i][j][k] += pre[i-1][j][k]
                pre[i][j][k] += pre[i][j-1][k]
                pre[i][j][k] += pre[i][j][k-1]
                pre[i][j][k] -= pre[i-1][j-1][k]
                pre[i][j][k] -= pre[i-1][j][k-1]
                pre[i][j][k] -= pre[i][j-1][k-1]
                pre[i][j][k] += pre[i-1][j-1][k-1]
    
    q = int(sys.stdin.readline())
    for _ in range(q):
        lx, rx, ly, ry, lz, rz = map(int, sys.stdin.readline().split())
        total = pre[rx][ry][rz]
        total -= pre[lx-1][ry][rz]
        total -= pre[rx][ly-1][rz]
        total -= pre[rx][ry][lz-1]
        total += pre[lx-1][ly-1][rz]
        total += pre[rx][ly-1][lz]
        total += pre[lx-1][ry][lz]
        total -= pre[lx-1][ly-1][lz]
        print(total)

if __name__ == "__main__":
    main()