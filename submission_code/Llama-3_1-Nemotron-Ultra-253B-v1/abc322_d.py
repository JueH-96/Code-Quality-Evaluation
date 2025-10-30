def parse_poly(lines):
    cells = []
    for r in range(4):
        for c in range(4):
            if lines[r][c] == '#':
                cells.append((r, c))
    return cells

def generate_rotations(original):
    rotations = []
    rot0 = original
    rot90 = [(y, -x) for (x, y) in original]
    rot180 = [(-x, -y) for (x, y) in original]
    rot270 = [(-y, x) for (x, y) in original]
    rotated_shapes = []
    for rotated in [rot0, rot90, rot180, rot270]:
        min_x = min(r[0] for r in rotated)
        min_y = min(r[1] for r in rotated)
        shifted = [(x - min_x, y - min_y) for (x, y) in rotated]
        rotated_shapes.append(frozenset(shifted))
    seen = set()
    unique = []
    for s in rotated_shapes:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

poly1 = [input().strip() for _ in range(4)]
poly2 = [input().strip() for _ in range(4)]
poly3 = [input().strip() for _ in range(4)]

cells1 = parse_poly(poly1)
cells2 = parse_poly(poly2)
cells3 = parse_poly(poly3)

total = len(cells1) + len(cells2) + len(cells3)
if total != 16:
    print("No")
    exit()

rotated1 = generate_rotations(cells1)
rotated2 = generate_rotations(cells2)
rotated3 = generate_rotations(cells3)

def get_positions(shape):
    if not shape:
        return []
    xs = [x for x, y in shape]
    ys = [y for x, y in shape]
    max_x = max(xs)
    max_y = max(ys)
    h = max_x + 1
    w = max_y + 1
    positions = []
    for i in range(4 - h + 1):
        for j in range(4 - w + 1):
            positions.append((i, j))
    return positions

rotated_info1 = [(s, get_positions(s)) for s in rotated1]
rotated_info2 = [(s, get_positions(s)) for s in rotated2]
rotated_info3 = [(s, get_positions(s)) for s in rotated3]

for s1, pos1 in rotated_info1:
    for s2, pos2 in rotated_info2:
        for s3, pos3 in rotated_info3:
            for i1, j1 in pos1:
                cells_s1 = {(i1 + x, j1 + y) for x, y in s1}
                for i2, j2 in pos2:
                    cells_s2 = {(i2 + x, j2 + y) for x, y in s2}
                    if cells_s1 & cells_s2:
                        continue
                    combined = cells_s1 | cells_s2
                    if len(combined) != len(s1) + len(s2):
                        continue
                    for i3, j3 in pos3:
                        cells_s3 = {(i3 + x, j3 + y) for x, y in s3}
                        if combined & cells_s3:
                            continue
                        if len(combined | cells_s3) == 16:
                            print("Yes")
                            exit()

print("No")