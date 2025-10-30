import sys


def main() -> None:
    sys.setrecursionlimit(1_000_000)

    # read first line
    it = iter(sys.stdin.read().split())
    H = int(next(it))
    W = int(next(it))
    N = int(next(it))

    # store holes by row
    row_holes = [set() for _ in range(H + 1)]          # 1-indexed
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        row_holes[a].add(b)

    prev = [0] * (W + 1)          # dp of the previous row
    total = 0

    for r in range(1, H + 1):
        holes = row_holes[r]      # columns that have holes in this row
        curr = [0] * (W + 1)      # dp for the current row

        # j = 0 is dummy, real columns are 1..W
        if holes:
            for c in range(1, W + 1):
                if c not in holes:                       # the cell is hole-less
                    m = prev[c]
                    if curr[c - 1] < m:
                        m = curr[c - 1]
                    t = prev[c - 1]
                    if t < m:
                        m = t
                    v = m + 1
                    curr[c] = v
                    total += v
                # else the cell is holed → curr[c] stays 0
        else:   # slightly faster when the whole row has no holes
            for c in range(1, W + 1):
                m = prev[c]
                if curr[c - 1] < m:
                    m = curr[c - 1]
                t = prev[c - 1]
                if t < m:
                    m = t
                v = m + 1
                curr[c] = v
                total += v

        prev = curr              # proceed to next row

    print(total)


if __name__ == "__main__":
    main()