def main():
    import sys
    input = sys.stdin.readline

    # Read the three shapes, each as 4 lines of 4 chars
    shapes = []
    for _ in range(3):
        shape = [input().rstrip('
') for __ in range(4)]
        shapes.append(shape)

    # Extract cell positions and areas
    all_cells = []
    areas = []
    for shape in shapes:
        cells = [(r, c)
                 for r in range(4) for c in range(4)
                 if shape[r][c] == '#']
        all_cells.append(cells)
        areas.append(len(cells))

    # Quick area check: must sum to 16
    if sum(areas) != 16:
        print("No")
        return

    # Generate all rotations (no flips) of a shape, normalized so min r,c = 0
    def get_rotations(cells):
        rots = []
        seen = set()
        cur = cells[:]
        for _ in range(4):
            # normalize
            minr = min(r for r,c in cur)
            minc = min(c for r,c in cur)
            norm = tuple(sorted(((r-minr, c-minc) for r,c in cur)))
            fr = frozenset(norm)
            if fr not in seen:
                seen.add(fr)
                rots.append(norm)
            # rotate 90° CW about origin: (r,c) -> (c, -r)
            cur = [(c, -r) for (r, c) in cur]
        return rots

    # For a normalized shape (list of (r,c) with min=0), generate all placement bitmasks in 4x4
    def placements_of(norm_shape):
        # compute bounding box
        maxr = max(r for r,c in norm_shape)
        maxc = max(c for r,c in norm_shape)
        h = maxr + 1
        w = maxc + 1
        masks = set()
        for dr in range(4 - h + 1):
            for dc in range(4 - w + 1):
                m = 0
                for (r, c) in norm_shape:
                    rr = r + dr
                    cc = c + dc
                    bit = rr * 4 + cc
                    m |= 1 << bit
                masks.add(m)
        return masks

    # Build all placement masks for each shape
    shape_placements = []
    for cells in all_cells:
        rots = get_rotations(cells)
        masks = set()
        for norm in rots:
            masks |= placements_of(norm)
        shape_placements.append(masks)

    # Sort shapes by number of placements (smallest first) to prune search
    shape_placements.sort(key=len)
    A, B, C = shape_placements

    FULL = (1 << 16) - 1
    # Try all combinations
    for a in A:
        for b in B:
            if a & b:
                continue
            ab = a | b
            for c in C:
                if ab & c:
                    continue
                if (ab | c) == FULL:
                    print("Yes")
                    return

    print("No")


if __name__ == "__main__":
    main()