from itertools import permutations

def normalize(points):
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    return [(x - min_x, y - min_y) for (x, y) in points]

def generate_rotations(shape):
    rotations = []
    current = shape.copy()
    for _ in range(4):
        normalized = normalize(current)
        rotations.append(normalized)
        # Rotate 90 degrees clockwise
        current = [(y, -x) for (x, y) in current]
    # Remove duplicates
    seen = set()
    unique = []
    for r in rotations:
        key = frozenset(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

poly = []
for _ in range(3):
    grid = []
    for _ in range(4):
        line = input().strip()
        grid.append(line)
    cells = []
    for i in range(4):
        for j in range(4):
            if grid[i][j] == '#':
                cells.append((i, j))
    poly.append(cells)

total_cells = sum(len(p) for p in poly)
if total_cells != 16:
    print("No")
    exit()

all_placements = []
for p in poly:
    placements = []
    rotations = generate_rotations(p)
    for rot in rotations:
        if not rot:
            continue
        max_dx = max(x for x, y in rot)
        max_dy = max(y for x, y in rot)
        for x in range(0, 4 - max_dx):
            for y in range(0, 4 - max_dy):
                cells = set()
                valid = True
                for dx, dy in rot:
                    cell_x = x + dx
                    cell_y = y + dy
                    if 0 <= cell_x < 4 and 0 <= cell_y < 4:
                        cells.add((cell_x, cell_y))
                    else:
                        valid = False
                        break
                if valid and len(cells) == len(rot):
                    placements.append(cells)
    all_placements.append(placements)

found = False
for perm in permutations([0, 1, 2]):
    a, b, c = perm
    for p1 in all_placements[a]:
        for p2 in all_placements[b]:
            if p1.intersection(p2):
                continue
            combined = p1.union(p2)
            for p3 in all_placements[c]:
                if combined.intersection(p3):
                    continue
                total = combined.union(p3)
                if len(total) == 16:
                    print("Yes")
                    exit()

print("No")