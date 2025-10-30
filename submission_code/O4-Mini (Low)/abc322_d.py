def main():
    import sys
    data = sys.stdin.read().splitlines()
    # Read three 4x4 shapes
    shapes = []
    for i in range(3):
        grid = data[4*i:4*i+4]
        cells = [(r, c) for r in range(4) for c in range(4) if grid[r][c] == '#']
        shapes.append(cells)

    # Generate all rotation variants (no flips) of a shape, normalized to top-left origin
    def variants(cells):
        vars_set = set()
        for rot in range(4):
            pts = []
            for (r, c) in cells:
                x, y = c, r
                # apply rot times 90° clockwise
                for _ in range(rot):
                    x, y = y, -x
                pts.append((y, x))
            # normalize to min row,col = 0
            min_r = min(p[0] for p in pts)
            min_c = min(p[1] for p in pts)
            norm = tuple(sorted(((p[0]-min_r, p[1]-min_c) for p in pts)))
            vars_set.add(norm)
        return [list(v) for v in vars_set]

    # For each shape generate all possible placements as 16-bit masks
    all_places = []
    FULL = (1<<16) - 1
    for cells in shapes:
        vars = variants(cells)
        placements = []
        for var in vars:
            max_r = max(r for r, c in var)
            max_c = max(c for r, c in var)
            h = max_r + 1
            w = max_c + 1
            # slide within 4x4
            for dr in range(4 - h + 1):
                for dc in range(4 - w + 1):
                    m = 0
                    for (r, c) in var:
                        rr = r + dr
                        cc = c + dc
                        bit = rr*4 + cc
                        m |= 1 << bit
                    placements.append(m)
        all_places.append(placements)

    p0, p1, p2 = all_places
    # Try all non-overlapping combinations that cover full grid
    for a in p0:
        for b in p1:
            if a & b: continue
            ab = a | b
            for c in p2:
                if ab & c: continue
                if (ab | c) == FULL:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()