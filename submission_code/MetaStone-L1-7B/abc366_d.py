def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1

    # Initialize A as a 3D list with 1-based indices
    A = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            start = (y - 1) * N
            for z in range(1, N + 1):
                idx = start + (z - 1)
                A[x][y][z] = int(input[ptr])
                ptr += 1

    # Initialize prefix sum array
    prefix = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                current = A[x][y][z]
                current += prefix[x-1][y][z]
                current += prefix[x][y-1][z]
                current += prefix[x][y][z-1]
                current -= prefix[x-1][y-1][z]
                current -= prefix[x-1][y][z-1]
                current -= prefix[x][y-1][z-1]
                current += prefix[x-1][y-1][z-1]
                prefix[x][y][z] = current

    Q = int(input[ptr])
    ptr += 1

    for _ in range(Q):
        Lx = int(input[ptr])
        Rx = int(input[ptr + 1])
        Ly = int(input[ptr + 2])
        Ry = int(input[ptr + 3])
        Lz = int(input[ptr + 4])
        Rz = int(input[ptr + 5])
        ptr += 6

        x1 = Lx
        x2 = Rx
        y1 = Ly
        y2 = Ry
        z1 = Lz
        z2 = Rz

        sum_val = prefix[x2][y2][z2]
        sum_val -= prefix[x1 - 1][y2][z2]
        sum_val -= prefix[x2][y1 - 1][z2]
        sum_val -= prefix[x2][y2][z1 - 1]
        sum_val += prefix[x1 - 1][y1 - 1][z2]
        sum_val += prefix[x1 - 1][y2][z1 - 1]
        sum_val += prefix[x2][y1 - 1][z1 - 1]
        sum_val -= prefix[x1 - 1][y1 - 1][z1 - 1]

        print(sum_val)

if __name__ == '__main__':
    main()