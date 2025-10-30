import sys

def main():
    lines = [sys.stdin.readline().strip() for _ in range(12)]
    polys = []
    for i in range(3):
        start = i * 4
        poly_lines = lines[start:start+4]
        initial_coords = []
        for j in range(4):
            for k in range(4):
                if poly_lines[j][k] == '#':
                    initial_coords.append( (j, k) )
        polys.append(initial_coords)
    # Check sum of cells
    sum_size = sum( len(p) for p in polys )
    if sum_size != 16:
        print("No")
        return
    # Helper functions
    def normalize_coords(coords):
        min_x = min(x for x, y in coords)
        min_y = min(y for x, y in coords)
        normalized = [ (x - min_x, y - min_y) for x, y in coords ]
        return sorted(normalized)
    def rotate90(coords):
        return [ (y, -x) for x, y in coords ]
    def rotate_and_normalize(coords):
        rotated = rotate90(coords)
        return normalize_coords(rotated)
    # Generate masks for each poly
    def generate_masks(initial_coords):
        # Generate all orientations
        initial = normalize_coords(initial_coords)
        orientations = set()
        current = initial
        orientations.add(tuple(current))
        for _ in range(3):
            current = rotate_and_normalize(current)
            orientations.add(tuple(current))
        # Now generate masks for each orientation
        masks = []
        for orientation in orientations:
            # orientation is a list of tuples
            max_x = max(x for x, y in orientation)
            max_y = max(y for x, y in orientation)
            w = max_x + 1
            h = max_y + 1
            max_dx = 4 - w
            max_dy = 4 - h
            if max_dx < 0 or max_dy < 0:
                continue
            for dx in range(0, max_dx + 1):
                for dy in range(0, max_dy + 1):
                    placed_cells = [ (x + dx, y + dy) for x, y in orientation ]
                    bitmask = 0
                    for r, c in placed_cells:
                        bitmask |= (1 << (r * 4 + c))
                    masks.append(bitmask)
        return masks
    # Generate masks for each poly
    masks1 = generate_masks(polys[0])
    masks2 = generate_masks(polys[1])
    masks3 = generate_masks(polys[2])
    mask_set3 = set(masks3)
    # Check all pairs from masks1 and masks2
    for a in masks1:
        for b in masks2:
            if (a & b) == 0:
                combined = a | b
                required = (1 << 16) - 1 - combined
                if required in mask_set3:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()