def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    
    # Initialize A
    A = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                A[x][y][z] = int(data[ptr])
                ptr += 1
    
    # Compute prefix sums
    pre = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                pre[x][y][z] = A[x][y][z]
                pre[x][y][z] += pre[x-1][y][z]
                pre[x][y][z] += pre[x][y-1][z]
                pre[x][y][z] += pre[x][y][z-1]
                pre[x][y][z] -= pre[x-1][y-1][z]
                pre[x][y][z] -= pre[x-1][y][z-1]
                pre[x][y][z] -= pre[x][y-1][z-1]
                pre[x][y][z] += pre[x-1][y-1][z-1]
    
    # Process queries
    Q = int(data[ptr])
    ptr +=1
    output = []
    for _ in range(Q):
        Lx = int(data[ptr])
        Rx = int(data[ptr+1])
        Ly = int(data[ptr+2])
        Ry = int(data[ptr+3])
        Lz = int(data[ptr+4])
        Rz = int(data[ptr+5])
        ptr +=6
        
        a = pre[Rx][Ry][Rz]
        b = pre[Lx-1][Ry][Rz]
        c = pre[Rx][Ly-1][Rz]
        d = pre[Rx][Ry][Lz-1]
        e = pre[Lx-1][Ly-1][Rz]
        f = pre[Lx-1][Ry][Lz-1]
        g = pre[Rx][Ly-1][Lz-1]
        h = pre[Lx-1][Ly-1][Lz-1]
        
        total = a - b - c - d + e + f + g - h
        output.append(str(total))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()