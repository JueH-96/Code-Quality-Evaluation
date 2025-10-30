def main():
    import sys
    input_data = sys.stdin.read().split()
    pos = 0

    # Read N
    N = int(input_data[pos])
    pos += 1

    # Read A array (1-based indexing)
    A = [[[0]*(N+1) for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                A[x][y][z] = int(input_data[pos])
                pos += 1

    # Build prefix sums P (1-based indexing)
    P = [[[0]*(N+1) for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                P[x][y][z] = (A[x][y][z]
                              + P[x-1][y][z]
                              + P[x][y-1][z]
                              + P[x][y][z-1]
                              - P[x-1][y-1][z]
                              - P[x-1][y][z-1]
                              - P[x][y-1][z-1]
                              + P[x-1][y-1][z-1])

    # Read Q
    Q = int(input_data[pos])
    pos += 1

    # Process queries
    output = []
    for _ in range(Q):
        Lx = int(input_data[pos]); Rx = int(input_data[pos+1])
        Ly = int(input_data[pos+2]); Ry = int(input_data[pos+3])
        Lz = int(input_data[pos+4]); Rz = int(input_data[pos+5])
        pos += 6

        ans = (P[Rx][Ry][Rz]
               - P[Lx-1][Ry][Rz]
               - P[Rx][Ly-1][Rz]
               - P[Rx][Ry][Lz-1]
               + P[Lx-1][Ly-1][Rz]
               + P[Lx-1][Ry][Lz-1]
               + P[Rx][Ly-1][Lz-1]
               - P[Lx-1][Ly-1][Lz-1])
        output.append(str(ans))

    print("
".join(output))

# Do not forget to call main()
if __name__ == "__main__":
    main()