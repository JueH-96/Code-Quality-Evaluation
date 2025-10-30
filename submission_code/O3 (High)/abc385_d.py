import sys
import bisect
from math import inf
from collections import defaultdict


def merge_intervals(lst):
    """return sorted list of disjoint intervals obtained from lst"""
    if not lst:
        return []
    lst.sort()
    merged = [list(lst[0])]
    for l, r in lst[1:]:
        if l <= merged[-1][1]:                     # overlap
            if r > merged[-1][1]:
                merged[-1][1] = r
        else:                                      # new interval
            merged.append([l, r])
    return [tuple(p) for p in merged]              # immutable tuples


def contains(intervals, v):
    """True  iff  v is inside one of the sorted disjoint intervals"""
    i = bisect.bisect_right(intervals, (v, inf)) - 1
    return i >= 0 and intervals[i][0] <= v <= intervals[i][1]


def main() -> None:
    it = iter(sys.stdin.read().strip().split())

    n = int(next(it))
    m = int(next(it))
    sx = int(next(it))
    sy = int(next(it))

    houses = [(int(next(it)), int(next(it))) for _ in range(n)]

    rows = defaultdict(list)          # y  -> list of (x1, x2)
    cols = defaultdict(list)          # x  -> list of (y1, y2)

    x, y = sx, sy
    for _ in range(m):
        d = next(it)
        c = int(next(it))

        if d == 'U':
            nx, ny = x, y + c
            lo, hi = y, ny
            cols[x].append((min(lo, hi), max(lo, hi)))
        elif d == 'D':
            nx, ny = x, y - c
            lo, hi = y, ny
            cols[x].append((min(lo, hi), max(lo, hi)))
        elif d == 'L':
            nx, ny = x - c, y
            lo, hi = x, nx
            rows[y].append((min(lo, hi), max(lo, hi)))
        else:  # 'R'
            nx, ny = x + c, y
            lo, hi = x, nx
            rows[y].append((min(lo, hi), max(lo, hi)))

        x, y = nx, ny

    final_x, final_y = x, y

    # merge all interval lists
    rows = {k: merge_intervals(v) for k, v in rows.items()}
    cols = {k: merge_intervals(v) for k, v in cols.items()}

    visited = 0
    for hx, hy in houses:
        if (hy in rows and contains(rows[hy], hx)) or \
           (hx in cols and contains(cols[hx], hy)):
            visited += 1

    print(f"{final_x} {final_y} {visited}")


if __name__ == "__main__":
    main()