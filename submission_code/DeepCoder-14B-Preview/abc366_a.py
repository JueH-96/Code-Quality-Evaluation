def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0

    N = int(input[ptr])
    ptr += 1

    # Initialize A with (N+1) x (N+1) x (N+1) dimensions (1-based)
    A = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]

    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                A[x][y][z] = int(input[ptr])
                ptr += 1

    # Initialize prefix sum array
    prefix = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]

    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                prefix[x][y][z] = A[x][y][z] + prefix[x-1][y][z] + prefix[x][y-1][z] + prefix[x][y][z-1] \
                                  - prefix[x-1][y-1][z] - prefix[x-1][y][z-1] - prefix[x][y-1][z-1] + prefix[x-1][y-1][z-1]

    Q = int(input[ptr])
    ptr += 1

    results = []
    for _ in range(Q):
        Lx = int(input[ptr])
        Rx = int(input[ptr+1])
        Ly = int(input[ptr+2])
        Ry = int(input[ptr+3])
        Lz = int(input[ptr+4])
        Rz = int(input[ptr+5])
        ptr += 6

        # Calculate the sum using the inclusion-exclusion principle
        sum_val = prefix[Rx][Ry][Rz]
        sum_val -= prefix[Lx-1][Ry][Rz]
        sum_val -= prefix[Rx][Ly-1][Rz]
        sum_val -= prefix[Rx][Ry][Lz-1]
        sum_val += prefix[Lx-1][Ly-1][Rz]
        sum_val += prefix[Lx-1][Ry][Lz-1]
        sum_val += prefix[Rx][Ly-1][Lz-1]
        sum_val -= prefix[Lx-1][Ly-1][Lz-1]

        results.append(sum_val)

    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()