import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    idx = 0
    n = data[idx]
    idx += 1

    # Initialize A as a 3D list with 1-based indices
    A = [[[0] * (n + 2) for _ in range(n + 2)] for __ in range(n + 2)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            for z in range(1, n + 1):
                A[x][y][z] = data[idx]
                idx += 1

    # Compute the 3D prefix sum S
    S = [[[0] * (n + 2) for _ in range(n + 2)] for __ in range(n + 2)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            for z in range(1, n + 1):
                S[x][y][z] = A[x][y][z] + S[x-1][y][z] + S[x][y-1][z] + S[x][y][z-1] \
                            - S[x-1][y-1][z] - S[x-1][y][z-1] - S[x][y-1][z-1] \
                            + S[x-1][y-1][z-1]

    Q = data[idx]
    idx += 1

    # Process each query
    output = []
    for _ in range(Q):
        Lx = data[idx]
        Rx = data[idx+1]
        Ly = data[idx+2]
        Ry = data[idx+3]
        Lz = data[idx+4]
        Rz = data[idx+5]
        idx +=6

        # Apply inclusion-exclusion formula
        sum_val = S[Rx][Ry][Rz] \
                  - S[Lx-1][Ry][Rz] \
                  - S[Rx][Ly-1][Rz] \
                  - S[Rx][Ry][Lz-1] \
                  + S[Lx-1][Ly-1][Rz] \
                  + S[Lx-1][Ry][Lz-1] \
                  + S[Rx][Ly-1][Lz-1] \
                  - S[Lx-1][Ly-1][Lz-1]
        output.append(str(sum_val))

    print('
'.join(output))

if __name__ == "__main__":
    main()