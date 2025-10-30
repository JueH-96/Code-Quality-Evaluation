import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    ptr = 0
    n = data[ptr]
    ptr += 1

    # Initialize 3D array A
    A = [[[0]*(n+1) for _ in range(n+1)] for __ in range(n+1)]
    for x in range(1, n+1):
        for y in range(1, n+1):
            for z in range(1, n+1):
                A[x][y][z] = data[ptr]
                ptr += 1

    # Compute 3D prefix sum P
    P = [[[0]*(n+1) for _ in range(n+1)] for __ in range(n+1)]
    for x in range(1, n+1):
        for y in range(1, n+1):
            for z in range(1, n+1):
                P[x][y][z] = A[x][y][z] + P[x-1][y][z] + P[x][y-1][z] + P[x][y][z-1] \
                            - P[x-1][y-1][z] - P[x-1][y][z-1] - P[x][y-1][z-1] \
                            + P[x-1][y-1][z-1]

    q = data[ptr]
    ptr += 1

    for _ in range(q):
        Lx = data[ptr]
        Rx = data[ptr+1]
        Ly = data[ptr+2]
        Ry = data[ptr+3]
        Lz = data[ptr+4]
        Rz = data[ptr+5]
        ptr += 6

        # Calculate using inclusion-exclusion formula
        res = P[Rx][Ry][Rz] - P[Lx-1][Ry][Rz] - P[Rx][Ly-1][Rz] - P[Rx][Ry][Lz-1] \
            + P[Lx-1][Ly-1][Rz] + P[Lx-1][Ry][Lz-1] + P[Rx][Ly-1][Lz-1] \
            - P[Lx-1][Ly-1][Lz-1]
        print(res)

if __name__ == '__main__':
    main()