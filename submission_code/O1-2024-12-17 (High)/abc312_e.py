def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    cuboids = [tuple(map(int,input().split())) for _ in range(N)]
    # cuboids[i] = (x1, y1, z1, x2, y2, z2)

    # We want, for each cuboid i, the count of other cuboids j
    # that share a face (positive 2D area on a common boundary).

    # Key facts and strategy:
    # 1) The coordinates are all between 0..100 (inclusive for x2,y2,z2).
    # 2) No two cuboids overlap in volume, but they may share faces/edges/points.
    # 3) Sharing a face in the x-dimension means one has x2=c and the other x1=c,
    #    and their projections onto (y,z) overlap with positive area.
    #    Similarly for y-dimension c and z-dimension c.
    # 4) Because no volume overlaps, within the same “group” (e.g., all with x2=c),
    #    their (y,z)-rectangles are interior-disjoint, so the total covered area
    #    is at most 101*101=10,201 cells for that plane.  This allows a “grid fill”
    #    approach without excessive overhead.
    #
    # Outline:
    #   - Build lists of “rectangles” for each potential boundary plane:
    #       x2plane[c] = [(cuboid_id, y1,y2, z1,z2), ...]  for all cuboids with x2=c
    #       x1plane[c] = [(cuboid_id, y1,y2, z1,z2), ...]  for all cuboids with x1=c
    #     Similarly y2plane[c], y1plane[c], z2plane[c], z1plane[c].
    #
    #   - For each dimension (x, y, z), and each plane c=1..99, do:
    #       A = group of cuboids that end at c (e.g., x2=c)
    #       B = group of cuboids that start at c (e.g., x1=c)
    #       If either A or B is empty, skip.
    #
    #       Then we fill two 101x101 grids (1D array of length 101*101 to speed up):
    #         occupantA[...] = -1 initially
    #         occupantB[...] = -1 initially
    #       We mark occupantA[...] = cuboid_id over the area each A-cuboid covers
    #         (since they do not overlap in that plane, no double-assignment)
    #       Similarly occupantB[...] = cuboid_id for B cuboids.
    #       Then we scan all 101*101 cells; wherever occupantA[idx] != -1
    #         and occupantB[idx] != -1 (and they differ), we record adjacency
    #         in a set structure so we only count it once.
    #
    #   - Finally, for each cuboid i, we output the size of its adjacency set.
    #
    # This method is O(N) to build plane-lists + up to 99 planes * 3 dims * ~30k ops
    # per plane = under ~9 million operations, which is borderline but can work
    # in optimized Python if implemented carefully.

    MAXC = 101  # coordinate limit + 1

    # Build plane-index structures.  Each is a list of length MAXC,
    # storing a list of rectangles (id, ranges...) that lie on that boundary.
    x2plane = [[] for _ in range(MAXC)]
    x1plane = [[] for _ in range(MAXC)]
    y2plane = [[] for _ in range(MAXC)]
    y1plane = [[] for _ in range(MAXC)]
    z2plane = [[] for _ in range(MAXC)]
    z1plane = [[] for _ in range(MAXC)]

    for i,(x1,y1,z1,x2,y2,z2) in enumerate(cuboids):
        # For x-plane:
        x2plane[x2].append((i, y1, y2, z1, z2))  # This cuboid's "right" face at x2=c
        x1plane[x1].append((i, y1, y2, z1, z2))  # This cuboid's "left" face  at x1=c
        # For y-plane:
        y2plane[y2].append((i, x1, x2, z1, z2))
        y1plane[y1].append((i, x1, x2, z1, z2))
        # For z-plane:
        z2plane[z2].append((i, x1, x2, y1, y2))
        z1plane[z1].append((i, x1, x2, y1, y2))

    # Adjacency sets for each cuboid
    adj = [set() for _ in range(N)]

    # Helper routines to fill a 1D array (size=101*101) for occupant info
    
    # Fill occupant array in the plane "yz" => occupantArr[y*101 + z]
    def fill_occupant_yz(rects, occupantArr):
        # rects is a list of (cuboid_id, y1,y2, z1,z2)
        for (cid, y1, y2, z1, z2) in rects:
            for yy in range(y1, y2):
                base = yy*101
                for zz in range(z1, z2):
                    occupantArr[base + zz] = cid

    # Fill occupant array in plane "xz" => occupantArr[x*101 + z]
    def fill_occupant_xz(rects, occupantArr):
        # rects is a list of (cuboid_id, x1,x2, z1,z2)
        for (cid, x1, x2, z1, z2) in rects:
            for xx in range(x1, x2):
                base = xx*101
                for zz in range(z1, z2):
                    occupantArr[base + zz] = cid

    # Fill occupant array in plane "xy" => occupantArr[x*101 + y]
    def fill_occupant_xy(rects, occupantArr):
        # rects is a list of (cuboid_id, x1,x2, y1,y2)
        for (cid, x1, x2, y1, y2) in rects:
            for xx in range(x1, x2):
                base = xx*101
                for yy in range(y1, y2):
                    occupantArr[base + yy] = cid

    # Check collisions between two occupant arrays of length 101*101
    def check_collisions(occA, occB):
        for idx in range(101*101):
            a = occA[idx]
            if a != -1:
                b = occB[idx]
                if b != -1 and b != a:
                    adj[a].add(b)
                    adj[b].add(a)

    # Process all planes in each dimension, skipping c=0 or c=100
    # because they cannot produce a valid face-sharing partner within [0..100].
    # (x2=0 or x1=100 are impossible with x1<x2<=100)
    # We still do a loop from 1..99 (inclusive).
    
    # X-dimension planes
    for c in range(1,100):
        A = x2plane[c]
        B = x1plane[c]
        if len(A)==0 or len(B)==0:
            continue
        occA = [-1]*(101*101)
        occB = [-1]*(101*101)
        fill_occupant_yz(A, occA)
        fill_occupant_yz(B, occB)
        check_collisions(occA, occB)

    # Y-dimension planes
    for c in range(1,100):
        A = y2plane[c]
        B = y1plane[c]
        if len(A)==0 or len(B)==0:
            continue
        occA = [-1]*(101*101)
        occB = [-1]*(101*101)
        fill_occupant_xz(A, occA)
        fill_occupant_xz(B, occB)
        check_collisions(occA, occB)

    # Z-dimension planes
    for c in range(1,100):
        A = z2plane[c]
        B = z1plane[c]
        if len(A)==0 or len(B)==0:
            continue
        occA = [-1]*(101*101)
        occB = [-1]*(101*101)
        fill_occupant_xy(A, occA)
        fill_occupant_xy(B, occB)
        check_collisions(occA, occB)

    # Now output the result: for each cuboid i, how many distinct neighbors
    # share a face with it.
    for i in range(N):
        print(len(adj[i]))

# Do not forget to call main()
if __name__ == "__main__":
    main()