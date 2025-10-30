import sys
from collections import defaultdict
from bisect import bisect_right

def main():
    data = sys.stdin
    # Read N, M, starting coordinates
    first = data.readline().split()
    if not first:
        return
    N = int(first[0])
    M = int(first[1])
    Sx = int(first[2])
    Sy = int(first[3])

    # Read house positions
    houses = [tuple(map(int, data.readline().split())) for _ in range(N)]

    # Prepare to collect horizontal (row) and vertical (col) segments
    rowSegs = defaultdict(list)  # y -> list of (x_start, x_end)
    colSegs = defaultdict(list)  # x -> list of (y_start, y_end)

    # Track Santa's current position
    cx, cy = Sx, Sy

    # Read moves and build segment lists
    for _ in range(M):
        parts = data.readline().split()
        d = parts[0]
        c = int(parts[1])
        ox, oy = cx, cy
        if d == 'U':
            cy += c
            y1, y2 = oy, cy
            if y1 <= y2:
                colSegs[ox].append((y1, y2))
            else:
                colSegs[ox].append((y2, y1))
        elif d == 'D':
            cy -= c
            y1, y2 = oy, cy
            if y1 <= y2:
                colSegs[ox].append((y1, y2))
            else:
                colSegs[ox].append((y2, y1))
        elif d == 'R':
            cx += c
            x1, x2 = ox, cx
            if x1 <= x2:
                rowSegs[oy].append((x1, x2))
            else:
                rowSegs[oy].append((x2, x1))
        elif d == 'L':
            cx -= c
            x1, x2 = ox, cx
            if x1 <= x2:
                rowSegs[oy].append((x1, x2))
            else:
                rowSegs[oy].append((x2, x1))

    # Now cx, cy is Santa's final position

    # Merge intervals on each row
    for y in list(rowSegs.keys()):
        segs = rowSegs[y]
        segs.sort(key=lambda t: t[0])
        merged = []
        s0, e0 = segs[0]
        for s, e in segs[1:]:
            if s <= e0:
                # overlap or touch
                if e > e0:
                    e0 = e
            else:
                merged.append((s0, e0))
                s0, e0 = s, e
        merged.append((s0, e0))
        # store starts and ends separately for binary search
        starts = [iv[0] for iv in merged]
        ends   = [iv[1] for iv in merged]
        rowSegs[y] = (starts, ends)

    # Merge intervals on each column
    for x in list(colSegs.keys()):
        segs = colSegs[x]
        segs.sort(key=lambda t: t[0])
        merged = []
        s0, e0 = segs[0]
        for s, e in segs[1:]:
            if s <= e0:
                if e > e0:
                    e0 = e
            else:
                merged.append((s0, e0))
                s0, e0 = s, e
        merged.append((s0, e0))
        starts = [iv[0] for iv in merged]
        ends   = [iv[1] for iv in merged]
        colSegs[x] = (starts, ends)

    # Count distinct houses covered by any segment
    count = 0
    for x, y in houses:
        # Check horizontal coverage
        v = rowSegs.get(y)
        if v is not None:
            starts, ends = v
            idx = bisect_right(starts, x)
            if idx > 0 and x <= ends[idx-1]:
                count += 1
                continue
        # Check vertical coverage
        v = colSegs.get(x)
        if v is not None:
            starts, ends = v
            idx = bisect_right(starts, y)
            if idx > 0 and y <= ends[idx-1]:
                count += 1
                continue

    # Output final position and count
    print(cx, cy, count)

if __name__ == "__main__":
    main()