def main():
    import sys

    # Read all 12 lines (4 per shape)
    raw = [sys.stdin.readline().rstrip('
') for _ in range(12)]

    # Parse the shapes into three lists of 4 lines each
    shape_grids = [raw[0:4], raw[4:8], raw[8:12]]

    # Convert each 4x4 block into a list of relative coordinates for '#'
    def parse_shape(grid):
        coords = []
        for r in range(4):
            for c in range(4):
                if grid[r][c] == '#':
                    coords.append((r, c))
        return coords

    shapes = [parse_shape(sg) for sg in shape_grids]

    # Quick check: if total number of occupied squares != 16, the answer is No
    total_squares = sum(len(s) for s in shapes)
    if total_squares != 16:
        print("No")
        return

    # Helper to recenter a set of coords so that min row and min col are 0
    def recenter(coords):
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        return [(r - min_r, c - min_c) for (r, c) in coords]

    # Rotate a set of coords 90 degrees clockwise around origin (0,0)
    # Then recenter so the shape's top-left is (0,0)
    def rotate90(coords):
        # (r, c) -> (c, -r) in a standard mathematical sense
        # but then we will recenter
        rotated = [(c, -r) for (r, c) in coords]
        return recenter(rotated)

    # Convert a list of (r, c) into a bitmask for a 4x4 board
    def coords_to_bitmask(coords):
        bitmask = 0
        for (r, c) in coords:
            if not (0 <= r < 4 and 0 <= c < 4):
                # If it's out of 4x4 we won't use it directly here,
                # we'll create final bitmask after translation. 
                # So for now, assume these are 0-based coords that fit in bounding box.
                pass
            bit = 1 << (4*r + c)
            bitmask |= bit
        return bitmask

    # Generate all 4 (unflipped) rotations of a shape as top-left-based coordinates
    def generate_rotations(coords):
        rotations = []
        current = recenter(coords)
        for _ in range(4):
            rotations.append(current)
            current = rotate90(current)
        # In some rare cases, rotating might give duplicates, but that won't break correctness.
        # We'll just keep them all (duplicates are harmless).
        return rotations

    # For a shape in top-left-based coords, generate all possible 4x4 placements as bitmasks
    # We shift the shape so it fits inside a 4x4 region.
    def generate_placements(shape_coords):
        placements = []
        # Find bounding box for shape
        r_vals = [r for r, _ in shape_coords]
        c_vals = [c for _, c in shape_coords]
        max_r = max(r_vals)
        max_c = max(c_vals)

        # The shape can be translated so that the shape fits within 4x4
        # So row shift can go from 0..(3 - max_r), col shift from 0..(3 - max_c).
        row_limit = 4 - (max_r + 1)
        col_limit = 4 - (max_c + 1)

        for row_shift in range(row_limit + 1):
            for col_shift in range(col_limit + 1):
                bm = 0
                valid = True
                for (r, c) in shape_coords:
                    rr = r + row_shift
                    cc = c + col_shift
                    if not (0 <= rr < 4 and 0 <= cc < 4):
                        valid = False
                        break
                    bm |= (1 << (4*rr + cc))
                if valid:
                    placements.append(bm)
        return placements

    # Precompute all placements for each of the 3 shapes
    shape_placements = []
    for s in shapes:
        rot_list = generate_rotations(s)
        # For each rotation, generate placements
        all_bm = []
        for rot_shape in rot_list:
            all_bm.extend(generate_placements(rot_shape))
        # Deduplicate bitmasks (some rotations may coincide)
        all_bm = list(set(all_bm))
        # If this shape has no valid placement, no solution
        if not all_bm:
            print("No")
            return
        shape_placements.append(all_bm)

    # Now try all combinations of placements for shape1, shape2, shape3
    # We want to see if there's a triple (a, b, c) with no overlap and full coverage => 0xFFFF
    full_mask = (1 << 16) - 1

    A = shape_placements[0]
    B = shape_placements[1]
    C = shape_placements[2]

    for a in A:
        for b in B:
            if (a & b) != 0:
                continue
            ab = a | b
            for c in C:
                if (ab & c) != 0:
                    continue
                # Now coverage check
                if (ab | c) == full_mask:
                    print("Yes")
                    return
    # If none found
    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()