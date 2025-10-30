import sys
from itertools import product

def read_poly():
    shape = []
    for _ in range(4):
        line = sys.stdin.readline().strip()
        shape.extend(line)
    cells = [(i, j) for i in range(4) for j in range(4) if shape[i*4 + j] == '#']
    return cells

poly1 = read_poly()
poly2 = read_poly()
poly3 = read_poly()

def get_rotations(cells):
    rotations = []
    # 0 degrees
    normalized = normalize(cells.copy())
    if normalized not in rotations:
        rotations.append(normalized)
    # 90 degrees
    rotated = [(y, -x) for (x, y) in cells]
    normalized = normalize(rotated)
    if normalized not in rotations:
        rotations.append(normalized)
    # 180 degrees
    rotated = [(-x, -y) for (x, y) in cells]
    normalized = normalize(rotated)
    if normalized not in rotations:
        rotations.append(normalized)
    # 270 degrees
    rotated = [(-y, x) for (x, y) in cells]
    normalized = normalize(rotated)
    if normalized not in rotations:
        rotations.append(normalized)
    return rotations

def normalize(cells):
    if not cells:
        return []
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    normalized = [(x - min_x, y - min_y) for (x, y) in cells]
    return normalized

# Check sum of areas
area1 = len(poly1)
area2 = len(poly2)
area3 = len(poly3)
if area1 + area2 + area3 != 16:
    print("No")
    exit()

# Generate placements for each poly
poly1_placements = []
seen = set()
for shape in get_rotations(poly1):
    width = max(dx for (dx, dy) in shape) + 1
    height = max(dy for (dx, dy) in shape) + 1
    possible_x = 4 - width
    possible_y = 4 - height
    if possible_x < 0 or possible_y < 0:
        continue
    for x in range(possible_x + 1):
        for y in range(possible_y + 1):
            placement = set((x + dx, y + dy) for (dx, dy) in shape)
            key = frozenset(placement)
            if key not in seen:
                seen.add(key)
                poly1_placements.append(placement)

poly2_placements = []
seen = set()
for shape in get_rotations(poly2):
    width = max(dx for (dx, dy) in shape) + 1
    height = max(dy for (dx, dy) in shape) + 1
    possible_x = 4 - width
    possible_y = 4 - height
    if possible_x < 0 or possible_y < 0:
        continue
    for x in range(possible_x + 1):
        for y in range(possible_y + 1):
            placement = set((x + dx, y + dy) for (dx, dy) in shape)
            key = frozenset(placement)
            if key not in seen:
                seen.add(key)
                poly2_placements.append(placement)

poly3_placements = []
seen = set()
for shape in get_rotations(poly3):
    width = max(dx for (dx, dy) in shape) + 1
    height = max(dy for (dx, dy) in shape) + 1
    possible_x = 4 - width
    possible_y = 4 - height
    if possible_x < 0 or possible_y < 0:
        continue
    for x in range(possible_x + 1):
        for y in range(possible_y + 1):
            placement = set((x + dx, y + dy) for (dx, dy) in shape)
            key = frozenset(placement)
            if key not in seen:
                seen.add(key)
                poly3_placements.append(placement)

# Check all triplets
for p1 in poly1_placements:
    for p2 in poly2_placements:
        combined = p1.union(p2)
        if len(combined) == 16:
            for p3 in poly3_placements:
                combined = combined.union(p3)
                if len(combined) == 16:
                    print("Yes")
                    exit()

print("No")