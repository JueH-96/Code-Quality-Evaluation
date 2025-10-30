import sys

def rotate(shape, rot_id):
    size = 4
    if rot_id == 0:
        return frozenset((r, c) for r, c in shape)
    elif rot_id == 1:
        return frozenset((c, size - 1 - r) for r, c in shape)
    elif rot_id == 2:
        return frozenset((size - 1 - r, size - 1 - c) for r, c in shape)
    elif rot_id == 3:
        return frozenset((size - 1 - c, r) for r, c in shape)

data = sys.stdin.read().replace("
", "")
total_cells = 0
placements = [[] for _ in range(3)]

for i in range(3):
    start_idx = i * 16
    poly_str = data[start_idx : start_idx + 16]
    cells = []
    for pos in range(16):
        if poly_str[pos] == '#':
            r = pos // 4
            c = pos % 4
            cells.append((r, c))
    total_cells += len(cells)
    shape = frozenset(cells)
    rotated_shapes = set()
    for rot in range(4):
        rot_shape = rotate(shape, rot)
        rotated_shapes.add(rot_shape)
    for rot_shape in rotated_shapes:
        min_r = min(r for r, _ in rot_shape)
        max_r = max(r for r, _ in rot_shape)
        min_c = min(c for _, c in rot_shape)
        max_c = max(c for _, c in rot_shape)
        dy_min = -min_r
        dy_max = 3 - max_r
        dx_min = -min_c
        dx_max = 3 - max_c
        for dy in range(dy_min, dy_max + 1):
            for dx in range(dx_min, dx_max + 1):
                mask = 0
                for r, c in rot_shape:
                    nr = r + dy
                    nc = c + dx
                    idx = nr * 4 + nc
                    mask |= (1 << idx)
                placements[i].append(mask)

if total_cells != 16:
    print("No")
else:
    def can_place(idx, occ_mask):
        if idx == 3:
            return True
        for mask in placements[idx]:
            if (mask & occ_mask) == 0:
                if can_place(idx + 1, occ_mask | mask):
                    return True
        return False
    if can_place(0, 0):
        print("Yes")
    else:
        print("No")