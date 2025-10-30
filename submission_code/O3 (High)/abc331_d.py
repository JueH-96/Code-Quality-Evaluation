import sys


def build_prefix(pattern):
    """
    Build 2-D prefix–sum for one N×N tile.
    pref[r][c] … #black in rectangle
                 rows [0,r) , cols [0,c)
    """
    n = len(pattern)
    pref = [[0] * (n + 1) for _ in range(n + 1)]

    for r in range(n):
        row = pattern[r]
        pr_above = pref[r]          # alias for speed
        pr_here  = pref[r + 1]
        for c in range(n):
            black = 1 if row[c] == 'B' else 0
            pr_here[c + 1] = (pr_above[c + 1] +
                              pr_here[c]       -
                              pr_above[c]      +
                              black)
    return pref


def blacks_upto(x, y, n, pref, tile_total):
    """
    Number of black squares in rectangle (0,0) – (x,y)   (inclusive)
    If either coordinate is <0  → 0.
    """
    if x < 0 or y < 0:
        return 0

    rows = x + 1                      # how many rows in the rectangle
    cols = y + 1                      # how many columns in the rectangle

    qx, rx = divmod(rows, n)          # rows  = qx·N  +  rx
    qy, ry = divmod(cols, n)          # cols  = qy·N  +  ry

    res = qx * qy * tile_total                    # full N×N tiles
    res += qx * pref[n][ry]                       # full-row tiles + col remainder
    res += qy * pref[rx][n]                       # full-col tiles + row remainder
    res += pref[rx][ry]                           # corner remainder
    return res


def main() -> None:
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    N = int(next(it))
    Q = int(next(it))

    pattern = [next(it).decode() for _ in range(N)]

    pref = build_prefix(pattern)
    tile_total = pref[N][N]                       # blacks in one full tile

    out_lines = []
    for _ in range(Q):
        A = int(next(it)); B = int(next(it))
        C = int(next(it)); D = int(next(it))

        ans = (blacks_upto(C, D, N, pref, tile_total)
               - blacks_upto(A - 1, D, N, pref, tile_total)
               - blacks_upto(C, B - 1, N, pref, tile_total)
               + blacks_upto(A - 1, B - 1, N, pref, tile_total))
        out_lines.append(str(ans))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()