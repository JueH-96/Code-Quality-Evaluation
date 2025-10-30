import sys

# constant: length of one axis of the unit grid (0..99)
M = 100                  # number of unit cells per dimension
CELLS = M * M            # 10 000

# ----------------------------------------------------------------------
# helper that fills a grid with rectangle id's
# ----------------------------------------------------------------------
def paint(grid, idx_list, a1, a2, b1, b2):
    """
    grid       : 1-D list of length CELLS, initially filled with -1
    idx_list   : indices of cuboids whose rectangles have to be painted
    (a1,a2)    : functions  returning the interval of the first grid axis
    (b1,b2)    : same for the second grid axis
    cells that belong to several rectangles never occur because within one
    group rectangles are area-disjoint.
    """
    for i in idx_list:
        y1 = a1[i]; y2 = a2[i]
        z1 = b1[i]; z2 = b2[i]
        for y in range(y1, y2):
            base = y * M
            for z in range(z1, z2):
                grid[base + z] = i
    # nothing is returned, grid mutated in place


# ----------------------------------------------------------------------
# core routine that, for one coordinate plane, finds all touching pairs
# ----------------------------------------------------------------------
def process_plane(groupA, groupB, getA1, getA2, getB1, getB2, cnt):
    """
    groupA : list of indices whose 'positive' face lies on the plane
    groupB : list of indices whose 'negative' face lies on the plane
    getA1..getB2 : arrays (or callables) that yield interval borders
    cnt    : result list, incremented for both members of every new pair
    """
    if not groupA or not groupB:
        return

    # grids for the two groups, -1 means empty
    gridA = [-1] * CELLS
    gridB = [-1] * CELLS

    paint(gridA, groupA, getA1, getA2, getB1, getB2)
    paint(gridB, groupB, getA1, getA2, getB1, getB2)

    seen = set()               # deduplicate pairs produced by many cells

    for idx in range(CELLS):
        a = gridA[idx]
        if a == -1:
            continue
        b = gridB[idx]
        if b == -1:
            continue
        code = (a << 17) | b   # N ≤ 1e5 < 2^17, unique packing
        if code not in seen:
            seen.add(code)
            cnt[a] += 1
            cnt[b] += 1


# ----------------------------------------------------------------------
def main() -> None:
    sys.setrecursionlimit(1 << 25)
    it = iter(sys.stdin.buffer.read().split())
    n = int(next(it))

    x1 = [0] * n
    y1 = [0] * n
    z1 = [0] * n
    x2 = [0] * n
    y2 = [0] * n
    z2 = [0] * n

    # faces lying on every integer coordinate for three axes
    left_x  = [[] for _ in range(101)]
    right_x = [[] for _ in range(101)]
    low_y   = [[] for _ in range(101)]
    high_y  = [[] for _ in range(101)]
    back_z  = [[] for _ in range(101)]
    front_z = [[] for _ in range(101)]

    for i in range(n):
        X1 = int(next(it)); Y1 = int(next(it)); Z1 = int(next(it))
        X2 = int(next(it)); Y2 = int(next(it)); Z2 = int(next(it))

        x1[i] = X1; y1[i] = Y1; z1[i] = Z1
        x2[i] = X2; y2[i] = Y2; z2[i] = Z2

        left_x[X1].append(i)
        right_x[X2].append(i)

        low_y[Y1].append(i)
        high_y[Y2].append(i)

        back_z[Z1].append(i)
        front_z[Z2].append(i)

    ans = [0] * n

    # 1. planes perpendicular to X (YZ rectangles)
    for k in range(101):
        process_plane(right_x[k], left_x[k], y1, y2, z1, z2, ans)

    # 2. planes perpendicular to Y (XZ rectangles)
    for k in range(101):
        process_plane(high_y[k], low_y[k], x1, x2, z1, z2, ans)

    # 3. planes perpendicular to Z (XY rectangles)
    for k in range(101):
        process_plane(front_z[k], back_z[k], x1, x2, y1, y2, ans)

    sys.stdout.write('
'.join(map(str, ans)))


if __name__ == "__main__":
    main()