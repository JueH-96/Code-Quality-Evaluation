def solve():
    N = int(input())
    A = []
    for _ in range(N):
        row_yz = []
        for _ in range(N):
            row_yz.append(list(map(int, input().split())))
        A.append(row_yz)
    
    Q = int(input())
    for _ in range(Q):
        Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())
        Lx -= 1
        Rx -= 1
        Ly -= 1
        Ry -= 1
        Lz -= 1
        Rz -= 1
        
        total_sum = 0
        for x in range(Lx, Rx + 1):
            for y in range(Ly, Ry + 1):
                for z in range(Lz, Rz + 1):
                    total_sum += A[x][y][z]
        print(total_sum)

solve()