import sys
def main():
    import sys
    data = sys.stdin.buffer.readline()
    if not data:
        return
    N = int(data)
    # Build 3D prefix sum array P of size (N+1)^3
    P = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    read = sys.stdin.buffer.readline
    # Read A[x][y][z] and build prefix sums
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            row = read().split()  # bytes tokens
            # row[z-1] is A[x][y][z]
            for z in range(1, N + 1):
                a = int(row[z - 1])
                P[x][y][z] = (
                    a
                    + P[x - 1][y][z]
                    + P[x][y - 1][z]
                    + P[x][y][z - 1]
                    - P[x - 1][y - 1][z]
                    - P[x - 1][y][z - 1]
                    - P[x][y - 1][z - 1]
                    + P[x - 1][y - 1][z - 1]
                )
    # Process queries
    Q = int(read())
    out = []
    append = out.append
    for _ in range(Q):
        l1, r1, l2, r2, l3, r3 = map(int, read().split())
        # inclusion-exclusion on P
        res = (
            P[r1][r2][r3]
            - P[l1 - 1][r2][r3]
            - P[r1][l2 - 1][r3]
            - P[r1][r2][l3 - 1]
            + P[l1 - 1][l2 - 1][r3]
            + P[l1 - 1][r2][l3 - 1]
            + P[r1][l2 - 1][l3 - 1]
            - P[l1 - 1][l2 - 1][l3 - 1]
        )
        append(str(res))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()