import sys
import threading

def main():
    import sys
    import bisect
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M, sx, sy = map(int, input().split())
    houses = []  # list of dicts: {'x':..., 'y':..., 'row_idx':..., 'col_idx':...}
    row_map = {}  # y -> list of (x, hid)
    col_map = {}  # x -> list of (y, hid)
    for hid in range(N):
        x, y = map(int, input().split())
        houses.append({'x': x, 'y': y})
        row_map.setdefault(y, []).append((x, hid))
        col_map.setdefault(x, []).append((y, hid))

    class NextArray:
        __slots__ = ('parent', 'K', 'coords', 'hids')
        def __init__(self, coords, hids):
            # coords: sorted list of coordinate (x or y), hids: same length, house ids
            self.coords = coords
            self.hids = hids
            self.K = len(coords)
            # parent size K+1: parent[i] = i initially; parent[K]=K sentinel
            self.parent = list(range(self.K + 1))
        def find(self, x):
            # iterative with path compression
            parent = self.parent
            while parent[x] != x:
                # path compression step
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(self, a, b):
            pa = self.find(a)
            pb = self.find(b)
            if pa != pb:
                self.parent[pa] = pb

    # Build row structures
    rows = {}  # y -> NextArray for that row
    for y, lst in row_map.items():
        # lst: list of (x, hid)
        lst.sort(key=lambda t: t[0])
        xs = [t[0] for t in lst]
        hids = [t[1] for t in lst]
        na = NextArray(xs, hids)
        rows[y] = na
        # record each house's index in its row
        for idx, hid in enumerate(hids):
            houses[hid]['row_idx'] = idx

    # Build column structures
    cols = {}  # x -> NextArray for that column
    for x, lst in col_map.items():
        # lst: list of (y, hid)
        lst.sort(key=lambda t: t[0])
        ys = [t[0] for t in lst]
        hids = [t[1] for t in lst]
        na = NextArray(ys, hids)
        cols[x] = na
        for idx, hid in enumerate(hids):
            houses[hid]['col_idx'] = idx

    visited = [False] * N
    cnt = 0
    cx, cy = sx, sy

    for _ in range(M):
        line = input().split()
        D = line[0]
        C = int(line[1])
        nx, ny = cx, cy
        if D == 'U':
            ny = cy + C
            x = cx
            # vertical segment x fixed, y in [cy, ny]
            if x in cols:
                col = cols[x]
                low = cy
                high = ny
                # find first idx with y >= low
                idx = bisect.bisect_left(col.coords, low)
                idx = col.find(idx)
                # scan while within high
                while idx < col.K and col.coords[idx] <= high:
                    hid = col.hids[idx]
                    # mark visited globally if not yet
                    if not visited[hid]:
                        visited[hid] = True
                        cnt += 1
                    # remove this index from both col and its row
                    col.union(idx, idx + 1)
                    # also remove from row
                    row_y = houses[hid]['y']
                    row = rows[row_y]
                    ridx = houses[hid]['row_idx']
                    row.union(ridx, ridx + 1)
                    idx = col.find(idx)
        elif D == 'D':
            ny = cy - C
            x = cx
            if ny > cy:
                low, high = cy, ny
            else:
                low, high = ny, cy
            if x in cols:
                col = cols[x]
                idx = bisect.bisect_left(col.coords, low)
                idx = col.find(idx)
                while idx < col.K and col.coords[idx] <= high:
                    hid = col.hids[idx]
                    if not visited[hid]:
                        visited[hid] = True
                        cnt += 1
                    col.union(idx, idx + 1)
                    row_y = houses[hid]['y']
                    row = rows[row_y]
                    ridx = houses[hid]['row_idx']
                    row.union(ridx, ridx + 1)
                    idx = col.find(idx)
        elif D == 'R':
            nx = cx + C
            y = cy
            if y in rows:
                row = rows[y]
                low = cx
                high = nx
                idx = bisect.bisect_left(row.coords, low)
                idx = row.find(idx)
                while idx < row.K and row.coords[idx] <= high:
                    hid = row.hids[idx]
                    if not visited[hid]:
                        visited[hid] = True
                        cnt += 1
                    row.union(idx, idx + 1)
                    col_x = houses[hid]['x']
                    col = cols[col_x]
                    cidx = houses[hid]['col_idx']
                    col.union(cidx, cidx + 1)
                    idx = row.find(idx)
        else:  # 'L'
            nx = cx - C
            y = cy
            if nx > cx:
                low, high = cx, nx
            else:
                low, high = nx, cx
            if y in rows:
                row = rows[y]
                idx = bisect.bisect_left(row.coords, low)
                idx = row.find(idx)
                while idx < row.K and row.coords[idx] <= high:
                    hid = row.hids[idx]
                    if not visited[hid]:
                        visited[hid] = True
                        cnt += 1
                    row.union(idx, idx + 1)
                    col_x = houses[hid]['x']
                    col = cols[col_x]
                    cidx = houses[hid]['col_idx']
                    col.union(cidx, cidx + 1)
                    idx = row.find(idx)

        cx, cy = nx, ny

    # final position and count
    print(cx, cy, cnt)

if __name__ == "__main__":
    main()