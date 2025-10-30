def main():
    import sys

    # Read the three polyominoes
    poly = []
    for _ in range(3):
        part = []
        for i in range(4):
            line = sys.stdin.readline().strip()
            for j in range(4):
                if line[j] == '#':
                    part.append((i, j))
        poly.append(part)

    # Check if the total number of cells is 16
    total_cells = sum(len(p) for p in poly)
    if total_cells != 16:
        print("No")
        return

    # Function to rotate a shape 90 degrees clockwise 'times' times
    def rotate(shape, times):
        for _ in range(times):
            shape = [(y, 3 - x) for (x, y) in shape]
        return shape

    # Generate all unique rotations for a shape
    def generate_rotations(shape):
        rotations = set()
        for r in range(4):
            rotated = rotate(shape, r)
            rotations.add(frozenset(rotated))
        return [list(s) for s in rotations]

    # Generate all possible placements (rotated and translated) for a shape
    def generate_placements(shape):
        placements = set()
        rotated_shapes = generate_rotations(shape)
        for rs in rotated_shapes:
            # Calculate min and max coordinates
            x_coords = [x for x, y in rs]
            y_coords = [y for x, y in rs]
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # Determine valid dx and dy ranges
            dx_min = -min_x
            dx_max = 3 - max_x
            dy_min = -min_y
            dy_max = 3 - max_y

            # Generate all possible (dx, dy) positions
            for dx in range(dx_min, dx_max + 1):
                for dy in range(dy_min, dy_max + 1):
                    shifted = frozenset((x + dx, y + dy) for x, y in rs)
                    placements.add(shifted)
        return placements

    # Generate placements for each polyomino
    placements = []
    for p in poly:
        placements.append(generate_placements(p))

    # Check all combinations of placements
    for p1 in placements[0]:
        for p2 in placements[1]:
            for p3 in placements[2]:
                # Check if the union is exactly 16 cells with no overlaps
                if (p1 | p2 | p3) == ( frozenset(i,j) for i in range(4) for j in range(4) ) and
                    p1.isdisjoint(p2) and p1.isdisjoint(p3) and p2.isdisjoint(p3):
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()