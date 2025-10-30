import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1

    A = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                A[x][y][z] = int(data[ptr])
                ptr += 1

    prefix = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                prefix[x][y][z] = A[x][y][z] 
                + prefix[x-1][y][z] 
                + prefix[x][y-1][z] 
                + prefix[x][y][z-1] 
                - prefix[x-1][y-1][z] 
                - prefix[x-1][y][z-1] 
                - prefix[x][y-1][z-1] 
                + prefix[x-1][y-1][z-1]

    Q = int(data[ptr])
    ptr += 1
    res = []
    for _ in range(Q):
        Lx = int(data[ptr])
        Rx = int(data[ptr+1])
        Ly = int(data[ptr+2])
        Ry = int(data[ptr+3])
        Lz = int(data[ptr+4])
        Rz = int(data[ptr+5])
        ptr += 6

        sum_val = prefix[Rx][Ry][Rz] 
        - prefix[Lx-1][Ry][Rz] 
        - prefix[Rx][Ly-1][Rz] 
        - prefix[Rx][Ry][Lz-1] 
        + prefix[Lx-1][Ly-1][Rz] 
        + prefix[Lx-1][Ry][Lz-1] 
        + prefix[Rx][Ly-1][Lz-1] 
        - prefix[Lx-1][Ly-1][Lz-1]

        res.append(sum_val)

    print('
'.join(map(str, res)))

if __name__ == '__main__':
    main()