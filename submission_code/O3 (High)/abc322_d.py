import sys
from itertools import product


def rotate90(coords):
    # rotate 90° clockwise: (r, c) -> (c, -r)
    return {(c, -r) for r, c in coords}


def normalize(coords):
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def all_orientations(coords):
    """return list of unique orientations (after rotation, no flip)"""
    res = []
    seen = set()
    cur = coords
    for _ in range(4):
        norm = normalize(cur)
        key = tuple(sorted(norm))
        if key not in seen:
            seen.add(key)
            res.append(norm)
        cur = rotate90(cur)
    return res


def placements_in_grid(orientations):
    """return list of bitmasks of every placement of any orientation inside 4x4 grid"""
    full_list = []
    seen_masks = set()
    for shape in orientations:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        height = max_r + 1
        width = max_c + 1
        for dy in range(4 - height + 1):
            for dx in range(4 - width + 1):
                mask = 0
                valid = True
                for r, c in shape:
                    nr, nc = r + dy, c + dx
                    if not (0 <= nr < 4 and 0 <= nc < 4):
                        valid = False
                        break
                    mask |= 1 << (nr * 4 + nc)
                if valid and mask not in seen_masks:
                    seen_masks.add(mask)
                    full_list.append(mask)
    return full_list


def main():
    lines = [sys.stdin.readline().rstrip('
') for _ in range(12)]
    if len(lines) < 12:
        return
    poly_raw = [lines[i * 4:(i + 1) * 4] for i in range(3)]

    poly_coords = []
    total_area = 0
    for grid in poly_raw:
        coords = {(r, c) for r in range(4) for c in range(4) if grid[r][c] == '#'}
        poly_coords.append(coords)
        total_area += len(coords)

    # quick reject by area
    if total_area != 16:
        print("No")
        return

    # Pre–compute every placement for every polyomino
    placement_lists = []
    for coords in poly_coords:
        orientations = all_orientations(coords)
        placement_lists.append(placements_in_grid(orientations))

    # order shapes by number of placements to prune faster
    order = sorted(range(3), key=lambda i: len(placement_lists[i]))
    ordered_lists = [placement_lists[i] for i in order]

    full_mask = (1 << 16) - 1

    for m1 in ordered_lists[0]:
        for m2 in ordered_lists[1]:
            if m1 & m2:
                continue
            union12 = m1 | m2
            for m3 in ordered_lists[2]:
                if union12 & m3:
                    continue
                if (union12 | m3) == full_mask:
                    print("Yes")
                    return
    print("No")


if __name__ == "__main__":
    main()