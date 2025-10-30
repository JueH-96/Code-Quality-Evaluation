import sys
import threading
def main():
    import sys
    import bisect
    
    input = sys.stdin.readline
    N, M, sx, sy = map(int, input().split())
    houses = []
    # Read houses, store with indices
    for i in range(N):
        x, y = map(int, input().split())
        houses.append((x, y, i))
    # Build point maps for fast lookup
    dictVX_pts = {}  # x -> list of (y, id)
    dictHY_pts = {}  # y -> list of (x, id)
    for x, y, idx in houses:
        dictVX_pts.setdefault(x, []).append((y, idx))
        dictHY_pts.setdefault(y, []).append((x, idx))
    # Sort the point lists
    for x in dictVX_pts:
        dictVX_pts[x].sort()
    for y in dictHY_pts:
        dictHY_pts[y].sort()
    # Prepare interval collectors
    dictVX_int = {}  # x -> list of [y1,y2]
    dictHY_int = {}  # y -> list of [x1,x2]
    cx, cy = sx, sy
    # Read movements and record intervals
    for _ in range(M):
        d, c = input().split()
        c = int(c)
        if d == 'U':
            newx, newy = cx, cy + c
            lo, hi = cy, newy
            dictVX_int.setdefault(cx, []).append((lo, hi))
        elif d == 'D':
            newx, newy = cx, cy - c
            lo, hi = newy, cy
            dictVX_int.setdefault(cx, []).append((lo, hi))
        elif d == 'R':
            newx, newy = cx + c, cy
            lo, hi = cx, newx
            dictHY_int.setdefault(cy, []).append((lo, hi))
        elif d == 'L':
            newx, newy = cx - c, cy
            lo, hi = newx, cx
            dictHY_int.setdefault(cy, []).append((lo, hi))
        cx, cy = newx, newy
    # Now cx, cy is final position
    # Merge intervals and collect visited house ids
    visited = set()
    # Process vertical travels
    for x, intv in dictVX_int.items():
        if x not in dictVX_pts:
            continue
        # merge intervals
        intv.sort()
        merged = []
        cur_s, cur_e = intv[0]
        for s, e in intv[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        # for each merged interval, pick houses
        pts = dictVX_pts[x]  # sorted list of (y, id)
        ys = [p[0] for p in pts]
        for s, e in merged:
            l = bisect.bisect_left(ys, s)
            r = bisect.bisect_right(ys, e)
            for j in range(l, r):
                visited.add(pts[j][1])
    # Process horizontal travels
    for y, intv in dictHY_int.items():
        if y not in dictHY_pts:
            continue
        intv.sort()
        merged = []
        cur_s, cur_e = intv[0]
        for s, e in intv[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        pts = dictHY_pts[y]
        xs = [p[0] for p in pts]
        for s, e in merged:
            l = bisect.bisect_left(xs, s)
            r = bisect.bisect_right(xs, e)
            for j in range(l, r):
                visited.add(pts[j][1])
    # Output final position and count
    print(cx, cy, len(visited))

if __name__ == "__main__":
    main()