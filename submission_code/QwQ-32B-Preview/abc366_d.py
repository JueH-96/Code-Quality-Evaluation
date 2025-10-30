def main():
    import sys
    input = sys.stdin.read
    tokens = input().split()
    idx = 0

    N = int(tokens[idx])
    idx += 1

    a = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(N)]
    for x in range(N):
        for y in range(N):
            for z in range(N):
                a[x][y][z] = int(tokens[idx])
                idx += 1

    s = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(N+1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                s[x][y][z] = a[x-1][y-1][z-1] + s[x-1][y][z] + s[x][y-1][z] + s[x][y][z-1] - s[x-1][y-1][z] - s[x-1][y][z-1] - s[x][y-1][z-1] + s[x-1][y-1][z-1]

    Q = int(tokens[idx])
    idx += 1

    for _ in range(Q):
        Lx = int(tokens[idx])
        Rx = int(tokens[idx+1])
        Ly = int(tokens[idx+2])
        Ry = int(tokens[idx+3])
        Lz = int(tokens[idx+4])
        Rz = int(tokens[idx+5])
        idx += 6

        sum_val = s[Rx][Ry][Rz] - s[Lx-1][Ry][Rz] - s[Rx][Ly-1][Rz] - s[Rx][Ry][Lz-1] + s[Lx-1][Ly-1][Rz] + s[Lx-1][Ry][Lz-1] + s[Rx][Ly-1][Lz-1] - s[Lx-1][Ly-1][Lz-1]
        print(sum_val)

if __name__ == "__main__":
    main()