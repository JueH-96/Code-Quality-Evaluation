import sys


def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = iter(data)

    try:
        N = next(it)
    except StopIteration:
        return  # empty input

    SZ = N + 1                          # we use 1-based indices, 0th layer stays 0
    pref = [[[0] * SZ for _ in range(SZ)] for _ in range(SZ)]   # 3-D prefix sums

    # build the 3-D prefix sum table
    for x in range(1, SZ):
        for y in range(1, SZ):
            for z in range(1, SZ):
                v = next(it)
                pref[x][y][z] = (
                    v
                    + pref[x - 1][y][z]
                    + pref[x][y - 1][z]
                    + pref[x][y][z - 1]
                    - pref[x - 1][y - 1][z]
                    - pref[x - 1][y][z - 1]
                    - pref[x][y - 1][z - 1]
                    + pref[x - 1][y - 1][z - 1]
                )

    Q = next(it)
    out_lines = []
    p = pref  # local alias for speed

    for _ in range(Q):
        Lx = next(it)
        Rx = next(it)
        Ly = next(it)
        Ry = next(it)
        Lz = next(it)
        Rz = next(it)

        ans = (
            p[Rx][Ry][Rz]
            - p[Lx - 1][Ry][Rz]
            - p[Rx][Ly - 1][Rz]
            - p[Rx][Ry][Lz - 1]
            + p[Lx - 1][Ly - 1][Rz]
            + p[Lx - 1][Ry][Lz - 1]
            + p[Rx][Ly - 1][Lz - 1]
            - p[Lx - 1][Ly - 1][Lz - 1]
        )
        out_lines.append(str(ans))

    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()