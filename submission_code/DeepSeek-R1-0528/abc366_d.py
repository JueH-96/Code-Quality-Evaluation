import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    arr = [[[0] * N for _ in range(N)] for __ in range(N)]
    
    for x in range(N):
        for y in range(N):
            for z in range(N):
                arr[x][y][z] = int(next(it))
    
    pre = [[[0] * (N+1) for _ in range(N+1)] for __ in range(N+1)]
    
    for i in range(1, N+1):
        for j in range(1, N+1):
            for k in range(1, N+1):
                pre[i][j][k] = arr[i-1][j-1][k-1] + pre[i-1][j][k] + pre[i][j-1][k] + pre[i][j][k-1] \
                             - pre[i-1][j-1][k] - pre[i-1][j][k-1] - pre[i][j-1][k-1] \
                             + pre[i-1][j-1][k-1]
    
    Q = int(next(it))
    out_lines = []
    for _ in range(Q):
        Lx = int(next(it)); Rx = int(next(it))
        Ly = int(next(it)); Ry = int(next(it))
        Lz = int(next(it)); Rz = int(next(it))
        
        total = pre[Rx][Ry][Rz]
        m1 = pre[Lx-1][Ry][Rz]
        m2 = pre[Rx][Ly-1][Rz]
        m3 = pre[Rx][Ry][Lz-1]
        a1 = pre[Lx-1][Ly-1][Rz]
        a2 = pre[Lx-1][Ry][Lz-1]
        a3 = pre[Rx][Ly-1][Lz-1]
        m4 = pre[Lx-1][Ly-1][Lz-1]
        
        ans = total - m1 - m2 - m3 + a1 + a2 + a3 - m4
        out_lines.append(str(ans))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()