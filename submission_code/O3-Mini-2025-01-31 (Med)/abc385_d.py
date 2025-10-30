def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    Sx = int(next(it))
    Sy = int(next(it))
    
    # Read houses and store as list; also later we will need to know for each house
    # at which index it appears in the “group‐by‐y” and “group‐by‐x” lists.
    houses = [None] * N
    for i in range(N):
        x = int(next(it))
        y = int(next(it))
        houses[i] = (x, y)
    
    # We want to be able to quickly answer: “which houses lie on the horizontal line y0 
    # with x in [L,R]?” (for horizontal moves) and similarly on vertical segments.
    #
    # So we group houses by y (for horizontal queries) and by x (for vertical queries).
    # In each group, we keep a sorted list (by the coordinate that varies)
    # and we prepare a Fenwick tree that will allow us to “delete” a house 
    # once it is visited so that we do not re‐scan it in future queries.
    #
    # Additionally, since a house may appear in both groups, when we “visit” a house 
    # we want to remove it from both groups.
    
    from collections import defaultdict
    y_groups = defaultdict(list)
    x_groups = defaultdict(list)
    
    for i, (x, y) in enumerate(houses):
        y_groups[y].append((x, i))
        x_groups[x].append((y, i))
    
    # Sort each group by the coordinate that changes along the move.
    for y in y_groups:
        y_groups[y].sort(key=lambda pair: pair[0])
    for x in x_groups:
        x_groups[x].sort(key=lambda pair: pair[0])
    
    # Fenwick tree implementation (1-indexed internally, but external interface is 0-indexed).
    class Fenw:
        __slots__ = ('n', 'fw')
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n + 1)
        def build(self, arr):
            for i, val in enumerate(arr):
                self.fw[i+1] = val
            for i in range(1, self.n+1):
                j = i + (i & -i)
                if j <= self.n:
                    self.fw[j] += self.fw[i]
        def update(self, i, delta):
            # i is 0-indexed.
            i += 1
            while i <= self.n:
                self.fw[i] += delta
                i += i & -i
        def query(self, i):
            # sum from 0 up to i (inclusive)
            s = 0
            i += 1
            while i:
                s += self.fw[i]
                i -= i & -i
            return s
        def query_range(self, l, r):
            if l > r:
                return 0
            return self.query(r) - (self.query(l-1) if l > 0 else 0)
        def find_first(self, target):
            # Returns smallest index i (0-indexed) such that prefix sum >= target.
            idx = 0
            bit_mask = 1 << (self.n.bit_length() - 1)
            while bit_mask:
                next_idx = idx + bit_mask
                if next_idx <= self.n and self.fw[next_idx] < target:
                    target -= self.fw[next_idx]
                    idx = next_idx
                bit_mask //= 2
            return idx  # 0-indexed
    
    # For each group, we will store a structure containing:
    #   • 'coord': the list of x coordinates (for y-groups) or y coordinates (for x-groups)
    #   • 'ids': the corresponding house indices.
    #   • 'fenw': a Fenw tree initialized with 1 for every house (active).
    y_struct = {}
    for y, alist in y_groups.items():
        coords = [pair[0] for pair in alist]
        ids = [pair[1] for pair in alist]
        fenw = Fenw(len(coords))
        fenw.build([1] * len(coords))
        y_struct[y] = {'coord': coords, 'ids': ids, 'fenw': fenw}
    
    x_struct = {}
    for x, alist in x_groups.items():
        coords = [pair[0] for pair in alist]
        ids = [pair[1] for pair in alist]
        fenw = Fenw(len(coords))
        fenw.build([1] * len(coords))
        x_struct[x] = {'coord': coords, 'ids': ids, 'fenw': fenw}
    
    # For each house, record its index position in its y-group and its x-group.
    pos_in_y = [None] * N
    for y, info in y_struct.items():
        for pos, hid in enumerate(info['ids']):
            pos_in_y[hid] = pos
    pos_in_x = [None] * N
    for x, info in x_struct.items():
        for pos, hid in enumerate(info['ids']):
            pos_in_x[hid] = pos
    
    # visited flag for each house.
    visited = [False] * N
    ans_count = 0

    # Current position of Santa.
    curx, cury = Sx, Sy
    
    # For each move, we simulate the move and at the same time “collect” houses that lie on the segment.
    # In a horizontal move the y coordinate is constant and key = cury; in a vertical move the x coordinate is constant.
    # For a move, the “segment” is from (curx, cury) to (nx, ny). We take coordinate interval [low, high]
    # and use binary search on the corresponding group's sorted coordinate list.
    for _ in range(M):
        d = next(it)
        c = int(next(it))
        if d in ('L', 'R'):
            # horizontal move (y is constant)
            if d == 'L':
                nx = curx - c
            else:
                nx = curx + c
            low = curx if curx < nx else nx
            high = nx if nx > curx else curx
            if cury in y_struct:
                group = y_struct[cury]
                arr = group['coord']
                li = bisect.bisect_left(arr, low)
                ri = bisect.bisect_right(arr, high) - 1
                if li <= ri:
                    fenw = group['fenw']
                    base = fenw.query(li-1) if li > 0 else 0
                    # While there is at least one active house in the [li, ri] interval,
                    # find and remove it.
                    while fenw.query_range(li, ri) > 0:
                        pos = fenw.find_first(base + 1)
                        if pos > ri:
                            break
                        hid = group['ids'][pos]
                        # It is possible (if already removed from a previous move) that the house is visited.
                        if not visited[hid]:
                            visited[hid] = True
                            ans_count += 1
                            # Remove from current (y-) group.
                            fenw.update(pos, -1)
                            # Also remove the house from its x-group.
                            hx, hy = houses[hid]
                            if hx in x_struct:
                                fenw_x = x_struct[hx]['fenw']
                                pos_x = pos_in_x[hid]
                                fenw_x.update(pos_x, -1)
                        else:
                            # If already visited, update so that it is not reported again.
                            fenw.update(pos, -1)
                        base = fenw.query(li-1) if li > 0 else 0
            curx = nx
        else:
            # vertical move (x is constant)
            if d == 'U':
                ny = cury + c
            else:
                ny = cury - c
            low = cury if cury < ny else ny
            high = ny if ny > cury else cury
            if curx in x_struct:
                group = x_struct[curx]
                arr = group['coord']
                li = bisect.bisect_left(arr, low)
                ri = bisect.bisect_right(arr, high) - 1
                if li <= ri:
                    fenw = group['fenw']
                    base = fenw.query(li-1) if li > 0 else 0
                    while fenw.query_range(li, ri) > 0:
                        pos = fenw.find_first(base + 1)
                        if pos > ri:
                            break
                        hid = group['ids'][pos]
                        if not visited[hid]:
                            visited[hid] = True
                            ans_count += 1
                            fenw.update(pos, -1)
                            # Remove from y-group as well.
                            hx, hy = houses[hid]
                            if hy in y_struct:
                                fenw_y = y_struct[hy]['fenw']
                                pos_y = pos_in_y[hid]
                                fenw_y.update(pos_y, -1)
                        else:
                            fenw.update(pos, -1)
                        base = fenw.query(li-1) if li > 0 else 0
            cury = ny

    # Output the final position and the number of distinct houses visited.
    sys.stdout.write(f"{curx} {cury} {ans_count}
")


if __name__ == '__main__':
    main()