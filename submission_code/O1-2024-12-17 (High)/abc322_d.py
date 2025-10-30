def main():
    import sys

    # Read input for each of the 3 polyominoes
    shapes = []
    total_squares = 0
    for _ in range(3):
        s = set()
        for r in range(4):
            line = sys.stdin.readline().rstrip('
')
            for c, ch in enumerate(line):
                if ch == '#':
                    s.add((r, c))
        shapes.append(s)
        total_squares += len(s)

    # If total number of occupied squares isn't 16, impossible to fill 4x4 without overlap/gap
    if total_squares != 16:
        print("No")
        return

    # Normalize shape so its top-left corner is (0,0)
    def normalize(shape):
        min_r = min(r for (r, c) in shape)
        min_c = min(c for (r, c) in shape)
        return set((r - min_r, c - min_c) for (r, c) in shape)

    # Rotate shape 90 degrees clockwise about (0,0), then normalize
    def rotate90(shape):
        new_shape = set()
        for (r, c) in shape:
            # (r, c) -> (c, -r)
            new_shape.add((c, -r))
        return normalize(new_shape)

    # Generate up to 4 orientations (0,90,180,270) for a shape
    # Remove duplicates if shape is symmetric upon rotation
    def get_orientations(shape):
        shape = normalize(shape)
        orientations = []
        curr = shape
        for _ in range(4):
            orientations.append(curr)
            curr = rotate90(curr)
        # Deduplicate
        unique = set()
        result = []
        for ori in orientations:
            fr = frozenset(ori)
            if fr not in unique:
                unique.add(fr)
                result.append(ori)
        return result

    # For a given shape (set of squares), generate all valid placements
    # in the 4x4 grid as bitmasks.
    def generate_masks(shape):
        masks = set()
        for ori in get_orientations(shape):
            max_r = max(r for (r, c) in ori)
            max_c = max(c for (r, c) in ori)
            height = max_r + 1
            width = max_c + 1
            # Try all offsets where the shape can fit in a 4x4 grid
            for ro in range(4 - height + 1):
                for co in range(4 - width + 1):
                    mask = 0
                    for (r, c) in ori:
                        pos = (r + ro) * 4 + (c + co)
                        mask |= (1 << pos)
                    masks.add(mask)
        return list(masks)

    # Precompute all possible placements (bitmasks) for each polyomino
    all_masks = [generate_masks(s) for s in shapes]

    FULL_MASK = (1 << 16) - 1

    # Try all combinations of placements (one per shape)
    for m1 in all_masks[0]:
        for m2 in all_masks[1]:
            if m1 & m2:  # overlap check
                continue
            union12 = m1 | m2
            for m3 in all_masks[2]:
                if union12 & m3:  # overlap check
                    continue
                # Check if together they cover the entire 4x4 (16 squares)
                if (union12 | m3) == FULL_MASK:
                    print("Yes")
                    return

    # If no combination worked
    print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()