# YOUR CODE HERE
import sys
import itertools

def read_polyomino():
    shape = []
    for _ in range(4):
        line = sys.stdin.readline().strip()
        shape.append(line)
    cells = set()
    for y in range(4):
        for x in range(4):
            if shape[y][x] == '#':
                cells.add((x, y))
    return cells

def rotate_polyomino(polyomino):
    rotations = []
    for i in range(4):
        new_cells = set()
        for x, y in polyomino:
            if i == 0:   # 0 degrees
                nx, ny = x, y
            elif i == 1:  # 90 degrees
                nx, ny = -y, x
            elif i == 2:  # 180 degrees
                nx, ny = -x, -y
            elif i == 3:  # 270 degrees
                nx, ny = y, -x
            new_cells.add((nx, ny))
        # Normalize positions to start from (0, 0)
        min_x = min(x for x, y in new_cells)
        min_y = min(y for x, y in new_cells)
        normalized_cells = frozenset((x - min_x, y - min_y) for x, y in new_cells)
        if normalized_cells not in rotations:
            rotations.append(normalized_cells)
    return rotations

def generate_placements(rotations):
    placements = []
    for shape in rotations:
        max_x = max(x for x, y in shape)
        max_y = max(y for x, y in shape)
        for dx in range(4 - max_x):
            for dy in range(4 - max_y):
                occupied = 0
                for x, y in shape:
                    pos = (y + dy) * 4 + (x + dx)
                    occupied |= 1 << pos
                placements.append(occupied)
    return placements

def main():
    polyominoes = []
    total_cells = 0
    for _ in range(3):
        poly = read_polyomino()
        total_cells += len(poly)
        polyominoes.append(poly)
    if total_cells != 16:
        print("No")
        return

    all_placements = []
    for poly in polyominoes:
        rotations = rotate_polyomino(poly)
        placements = generate_placements(rotations)
        all_placements.append(placements)

    full_grid = (1 << 16) - 1  # 0xFFFF
    for p1 in all_placements[0]:
        for p2 in all_placements[1]:
            if p1 & p2 != 0:
                continue
            for p3 in all_placements[2]:
                if (p1 & p3 != 0) or (p2 & p3 != 0):
                    continue
                if (p1 | p2 | p3) == full_grid:
                    print("Yes")
                    return
    print("No")

if __name__ == '__main__':
    main()