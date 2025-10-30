def main():
    import sys
    data = sys.stdin.read().split()
    pos = 0

    # Read the dimension N.
    N = int(data[pos])
    pos += 1

    # Read the 3D matrix A.
    # The matrix is given in order such that for each x from 1 to N,
    # we have N lines corresponding to y=1..N and each containing N integers for z=1..N.
    A = [[[0] * N for _ in range(N)] for __ in range(N)]
    for x in range(N):
        for y in range(N):
            for z in range(N):
                A[x][y][z] = int(data[pos])
                pos += 1

    # Build a 3D prefix sum array P with dimensions (N+1) x (N+1) x (N+1).
    # P[i][j][k] will store the sum of all A[x][y][z] with 1 <= x <= i, 1 <= y <= j, 1 <= z <= k.
    # We use 1-indexing for P so that boundary handling is simple.
    P = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            for k in range(1, N + 1):
                P[i][j][k] = (A[i - 1][j - 1][k - 1] +
                              P[i - 1][j][k] +
                              P[i][j - 1][k] +
                              P[i][j][k - 1] -
                              P[i - 1][j - 1][k] -
                              P[i - 1][j][k - 1] -
                              P[i][j - 1][k - 1] +
                              P[i - 1][j - 1][k - 1])

    # Now process Q queries.
    Q = int(data[pos])
    pos += 1
    out_lines = []
    for _ in range(Q):
        Lx = int(data[pos]); pos += 1
        Rx = int(data[pos]); pos += 1
        Ly = int(data[pos]); pos += 1
        Ry = int(data[pos]); pos += 1
        Lz = int(data[pos]); pos += 1
        Rz = int(data[pos]); pos += 1

        # Use inclusion-exclusion to get the sum within the sub-cuboid.
        ans = (P[Rx][Ry][Rz] -
               P[Lx - 1][Ry][Rz] -
               P[Rx][Ly - 1][Rz] -
               P[Rx][Ry][Lz - 1] +
               P[Lx - 1][Ly - 1][Rz] +
               P[Lx - 1][Ry][Lz - 1] +
               P[Rx][Ly - 1][Lz - 1] -
               P[Lx - 1][Ly - 1][Lz - 1])
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()