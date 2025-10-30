def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1

    # Initialize A as a 3D array (N+1)x(N+1)x(N+1) with 1-based indices
    A = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            # Read N elements for z from 1 to N
            for z in range(1, N+1):
                A[x][y][z] = int(input[ptr])
                ptr += 1

    # Compute the 3D prefix sum array
    prefix = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                prefix[x][y][z] = (
                    A[x][y][z]
                    + prefix[x-1][y][z]
                    + prefix[x][y-1][z]
                    + prefix[x][y][z-1]
                    - prefix[x-1][y-1][z]
                    - prefix[x-1][y][z-1]
                    - prefix[x][y-1][z-1]
                    + prefix[x-1][y-1][z-1]
                )

    Q = int(input[ptr])
    ptr += 1
    for _ in range(Q):
        lx = int(input[ptr])
        rx = int(input[ptr+1])
        ly = int(input[ptr+2])
        ry = int(input[ptr+3])
        lz = int(input[ptr+4])
        rz = int(input[ptr+5])
        ptr +=6
        # Calculate using inclusion-exclusion principle
        total = (
            prefix[rx][ry][rz]
            - prefix[lx-1][ry][rz]
            - prefix[rx][ly-1][rz]
            - prefix[rx][ry][lz-1]
            + prefix[lx-1][ly-1][rz]
            + prefix[lx-1][ry][lz-1]
            + prefix[rx][ly-1][lz-1]
            - prefix[lx-1][ly-1][lz-1]
        )
        print(total)

if __name__ == "__main__":
    main()