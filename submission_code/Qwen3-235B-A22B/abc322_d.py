import sys
from itertools import permutations

def read_polyomino():
    cells = []
    for i in range(4):
        line = sys.stdin.readline().strip()
        for j in range(4):
            if line[j] == '#':
                cells.append((i, j))
    return cells

def generate_bitmasks(cells):
    unique_shapes = set()
    for rot in range(4):
        rotated = []
        for (x, y) in cells:
            if rot == 0:
                rx, ry = x, y
            elif rot == 1:
                rx, ry = y, -x
            elif rot == 2:
                rx, ry = -x, -y
            else:
                rx, ry = -y, x
            rotated.append((rx, ry))
        min_x = min(p[0] for p in rotated)
        min_y = min(p[1] for p in rotated)
        normalized = [(x - min_x, y - min_y) for (x, y) in rotated]
        frozen = frozenset(normalized)
        unique_shapes.add(frozen)
    
    bitmask_set = set()
    for shape in unique_shapes:
        max_x = max(x for x, y in shape)
        max_y = max(y for x, y in shape)
        dx_max = 3 - max_x
        dy_max = 3 - max_y
        if dx_max < 0 or dy_max < 0:
            continue
        for dx in range(dx_max + 1):
            for dy in range(dy_max + 1):
                mask = 0
                valid = True
                for (x, y) in shape:
                    r = x + dx
                    c = y + dy
                    if not (0 <= r < 4 and 0 <= c < 4):
                        valid = False
                        break
                    pos = r * 4 + c
                    mask |= (1 << pos)
                if valid:
                    bitmask_set.add(mask)
    return bitmask_set

def main():
    poly_cells = []
    total = 0
    for _ in range(3):
        cells = read_polyomino()
        cnt = len(cells)
        total += cnt
        poly_cells.append(cells)
    if total != 16:
        print("No")
        return
    
    set_list = []
    for cells in poly_cells:
        masks = generate_bitmasks(cells)
        if not masks:
            print("No")
            return
        set_list.append(masks)
    
    for perm in permutations(range(3)):
        a, b, c = perm
        set_a = set_list[a]
        set_b = set_list[b]
        set_c = set_list[c]
        for mask_a in set_a:
            for mask_b in set_b:
                if (mask_a & mask_b) != 0:
                    continue
                combined = mask_a | mask_b
                remaining = 0xffff & (~combined)
                if remaining in set_c:
                    print("Yes")
                    return
    print("No")

if __name__ == '__main__':
    main()