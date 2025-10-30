import sys
import threading
def main():
    import sys
    from array import array
    sys.setrecursionlimit(10**7)
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    H, W, Q = map(int, line)
    # Disjoint-set structures:
    # rowL_par[r]: array of size W+1, indices 0..W
    # rowR_par[r]: array of size W+2, indices 0..W+1
    # colU_par[c]: array of size H+1, indices 0..H
    # colD_par[c]: array of size H+2, indices 0..H+1
    rowL_par = [None] * (H+1)
    rowR_par = [None] * (H+1)
    for r in range(1, H+1):
        # left DSU: 0..W
        rowL_par[r] = array('i', range(W+1))
        # right DSU: 0..W+1
        rowR_par[r] = array('i', range(W+2))
    colU_par = [None] * (W+1)
    colD_par = [None] * (W+1)
    for c in range(1, W+1):
        # up DSU: 0..H
        colU_par[c] = array('i', range(H+1))
        # down DSU: 0..H+1
        colD_par[c] = array('i', range(H+2))
    # existence of walls
    exist = [ bytearray(b'\x01') * (W+1) for _ in range(H+1) ]
    # initial count
    count = H * W
    # DSU find and union functions
    def findL(r, x):
        par = rowL_par[r]
        # find root
        root = x
        while par[root] != root:
            root = par[root]
        # path compress
        y = x
        while par[y] != root:
            tmp = par[y]
            par[y] = root
            y = tmp
        return root
    def unionL(r, x, y):
        par = rowL_par[r]
        rx = x
        # find rx
        while par[rx] != rx:
            rx = par[rx]
        ry = y
        while par[ry] != ry:
            ry = par[ry]
        par[rx] = ry
    def findR(r, x):
        par = rowR_par[r]
        root = x
        while par[root] != root:
            root = par[root]
        y = x
        while par[y] != root:
            tmp = par[y]
            par[y] = root
            y = tmp
        return root
    def unionR(r, x, y):
        par = rowR_par[r]
        rx = x
        while par[rx] != rx:
            rx = par[rx]
        ry = y
        while par[ry] != ry:
            ry = par[ry]
        par[rx] = ry
    def findU(c, x):
        par = colU_par[c]
        root = x
        while par[root] != root:
            root = par[root]
        y = x
        while par[y] != root:
            tmp = par[y]
            par[y] = root
            y = tmp
        return root
    def unionU(c, x, y):
        par = colU_par[c]
        rx = x
        while par[rx] != rx:
            rx = par[rx]
        ry = y
        while par[ry] != ry:
            ry = par[ry]
        par[rx] = ry
    def findD(c, x):
        par = colD_par[c]
        root = x
        while par[root] != root:
            root = par[root]
        y = x
        while par[y] != root:
            tmp = par[y]
            par[y] = root
            y = tmp
        return root
    def unionD(c, x, y):
        par = colD_par[c]
        rx = x
        while par[rx] != rx:
            rx = par[rx]
        ry = y
        while par[ry] != ry:
            ry = par[ry]
        par[rx] = ry

    out_count = count
    for _ in range(Q):
        line = data.readline().split()
        if not line:
            break
        r = int(line[0])
        c = int(line[1])
        # collect to_remove list
        to_rem = []
        if exist[r][c]:
            to_rem.append((r, c))
        else:
            # up
            u = findU(c, r)
            if u > 0:
                to_rem.append((u, c))
            # down
            d = findD(c, r)
            if d <= H:
                to_rem.append((d, c))
            # left
            l = findL(r, c)
            if l > 0:
                to_rem.append((r, l))
            # right
            rr = findR(r, c)
            if rr <= W:
                to_rem.append((r, rr))
        # perform removals
        for (rrr, ccc) in to_rem:
            if exist[rrr][ccc]:
                exist[rrr][ccc] = 0
                out_count -= 1
                # remove in DSU
                # row
                unionL(rrr, ccc, ccc-1)
                unionR(rrr, ccc, ccc+1)
                # column
                unionU(ccc, rrr, rrr-1)
                unionD(ccc, rrr, rrr+1)
    # output remaining walls
    sys.stdout.write(str(out_count))

if __name__ == "__main__":
    main()