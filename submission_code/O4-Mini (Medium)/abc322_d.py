def main():
    import sys

    # Read the 12 lines of input
    lines = [sys.stdin.readline().rstrip('
') for _ in range(12)]
    # Parse into three shapes, each 4x4
    shapes = []
    for si in range(3):
        coords = []
        for r in range(4):
            line = lines[si*4 + r]
            for c, ch in enumerate(line):
                if ch == '#':
                    coords.append((r, c))
        shapes.append(coords)

    # Quick check: sum of cells must be 16 to fill 4x4 grid
    total_cells = sum(len(s) for s in shapes)
    if total_cells != 16:
        print("No")
        return

    # Function to normalize a list of (r,c) so min r,c go to (0,0)
    def normalize(coords):
        rs = [r for r, c in coords]
        cs = [c for r, c in coords]
        minr = min(rs)
        minc = min(cs)
        return sorted((r - minr, c - minc) for r, c in coords)

    # Function to rotate coords 90° CCW about origin: (r,c)->(-c,r)
    def rotate90(coords):
        return [(-c, r) for r, c in coords]

    # For each shape, build list of all distinct orientations (normalized)
    all_orients = []
    for coords in shapes:
        # start with normalized base coords
        base = normalize(coords)
        seen = set()
        orients = []
        curr = base
        for _ in range(4):
            normed = tuple(normalize(curr))
            if normed not in seen:
                seen.add(normed)
                orients.append(list(normed))
            # rotate for next iteration
            curr = rotate90(curr)
        all_orients.append(orients)

    # For each oriented shape, generate all placements in 4x4 grid as bitmasks
    placements = []
    FULL = (1 << 16) - 1
    for orients in all_orients:
        pl_list = []
        for shape in orients:
            # find bounding box to know how far we can slide
            maxr = max(r for r, c in shape)
            maxc = max(c for r, c in shape)
            # slide shape so it remains within 0..3 in both coords
            for dr in range(0, 4 - maxr):
                for dc in range(0, 4 - maxc):
                    mask = 0
                    for (r, c) in shape:
                        nr = r + dr
                        nc = c + dc
                        bit = 1 << (nr * 4 + nc)
                        mask |= bit
                    pl_list.append(mask)
        placements.append(pl_list)

    # Now try to pick one placement for each of the three shapes
    P0, P1, P2 = placements
    for m0 in P0:
        for m1 in P1:
            if m0 & m1:
                continue
            m01 = m0 | m1
            for m2 in P2:
                if (m01 & m2) == 0 and (m01 | m2) == FULL:
                    print("Yes")
                    return

    print("No")

if __name__ == "__main__":
    main()