import sys
from itertools import permutations

def generate_placements(cells_set):
    placements = set()
    for rotation in [0, 90, 180, 270]:
        if rotation == 0:
            rotated = cells_set
        elif rotation == 90:
            rotated = { (y, 3 - x) for (x, y) in cells_set }
        elif rotation == 180:
            rotated = { (3 - x, 3 - y) for (x, y) in cells_set }
        elif rotation == 270:
            rotated = { (3 - y, x) for (x, y) in cells_set }
        # Normalize the rotated points
        if not rotated:
            continue
        min_x = min(p[0] for p in rotated)
        min_y = min(p[1] for p in rotated)
        normalized = { (x - min_x, y - min_y) for (x, y) in rotated }
        # Calculate max dimensions for normalization
        max_x = max(p[0] for p in normalized) if normalized else 0
        max_y = max(p[1] for p in normalized) if normalized else 0
        # Determine possible offsets
        for dx in range(0, 4 - max_x):
            for dy in range(0, 4 - max_y):
                placed = frozenset( (x + dx, y + dy) for (x, y) in normalized )
                placements.add(placed)
    # Convert to a list of sets
    return [set(p) for p in placements]

def main():
    cells = [set(), set(), set()]
    for i in range(3):
        current = set()
        for j in range(4):
            line = input().strip()
            for k in range(4):
                if line[k] == '#':
                    current.add( (j, k) )
        cells[i] = current

    total = sum(len(c) for c in cells)
    if total != 16:
        print("No")
        return

    poly_placements = []
    for i in range(3):
        placements = generate_placements(cells[i])
        poly_placements.append(placements)

    for order in permutations([0, 1, 2]):
        i, j, k = order
        for a in poly_placements[i]:
            for b in poly_placements[j]:
                if a & b:
                    continue
                for c in poly_placements[k]:
                    if (a & c) or (b & c):
                        continue
                    if len(a | b | c) == 16:
                        print("Yes")
                        return

    print("No")

if __name__ == "__main__":
    main()