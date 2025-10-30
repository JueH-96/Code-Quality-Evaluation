import sys

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    # Read cuboid coordinates
    X1 = [0]*(n+1)
    Y1 = [0]*(n+1)
    Z1 = [0]*(n+1)
    X2 = [0]*(n+1)
    Y2 = [0]*(n+1)
    Z2 = [0]*(n+1)
    for i in range(1, n+1):
        X1[i] = int(next(it)); Y1[i] = int(next(it)); Z1[i] = int(next(it))
        X2[i] = int(next(it)); Y2[i] = int(next(it)); Z2[i] = int(next(it))

    # We'll fill a 100x100x100 grid of unit cells.
    # grid[idx] = cuboid id occupying that cell, or 0 if empty.
    # idx = x*100*100 + y*100 + z
    from array import array
    SIZE = 100*100*100
    grid = array('I', [0]) * SIZE

    # Fill the grid with cuboid ids
    for cid in range(1, n+1):
        x1, y1, z1 = X1[cid], Y1[cid], Z1[cid]
        x2, y2, z2 = X2[cid], Y2[cid], Z2[cid]
        # Sum of all volumes is <= 1e6, so this is OK.
        for x in range(x1, x2):
            base_x = x * 10000
            for y in range(y1, y2):
                base_xy = base_x + y * 100
                for z in range(z1, z2):
                    grid[base_xy + z] = cid

    # Prepare adjacency sets
    adj = [set() for _ in range(n+1)]

    # For each cell, check +x, +y, +z neighbor for a different cuboid id.
    # If so, they share a unit face => they share positive-area face.
    for x in range(100):
        base_x = x * 10000
        if x+1 < 100:
            base_xp = (x+1) * 10000
        else:
            base_xp = None
        for y in range(100):
            base_xy = base_x + y * 100
            if y+1 < 100:
                base_x_yp = base_x + (y+1)*100
            else:
                base_x_yp = None
            for z in range(100):
                cid = grid[base_xy + z]
                if cid == 0:
                    continue
                # x+1 neighbor
                if base_xp is not None:
                    cid2 = grid[base_xp + y*100 + z]
                    if cid2 != 0 and cid2 != cid:
                        adj[cid].add(cid2)
                        adj[cid2].add(cid)
                # y+1 neighbor
                if base_x_yp is not None:
                    cid2 = grid[base_x_yp + z]
                    if cid2 != 0 and cid2 != cid:
                        adj[cid].add(cid2)
                        adj[cid2].add(cid)
                # z+1 neighbor
                if z+1 < 100:
                    cid2 = grid[base_xy + (z+1)]
                    if cid2 != 0 and cid2 != cid:
                        adj[cid].add(cid2)
                        adj[cid2].add(cid)

    # Output result
    out = sys.stdout
    write = out.write
    for i in range(1, n+1):
        write(str(len(adj[i])) + "
")

if __name__ == "__main__":
    main()