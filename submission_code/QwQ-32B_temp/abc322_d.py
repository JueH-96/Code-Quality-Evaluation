import sys

def read_poly():
    cells = []
    for x in range(4):
        line = sys.stdin.readline().strip()
        for y in range(4):
            if line[y] == '#':
                cells.append((x, y))
    return cells

def generate_placements(cells):
    if not cells:
        return []
    min_x = min(x for x, y in cells)
    min_y = min(y for x, y in cells)
    relative = [(x - min_x, y - min_y) for x, y in cells]
    placements = set()
    
    for rot in 0, 1, 2, 3:
        rotated = []
        for dx, dy in relative:
            if rot == 0:
                new_dx, new_dy = dx, dy
            elif rot == 1:
                new_dx, new_dy = dy, -dx
            elif rot == 2:
                new_dx, new_dy = -dx, -dy
            elif rot == 3:
                new_dx, new_dy = -dy, dx
            rotated.append((new_dx, new_dy))
        
        min_dx = min(dx for dx, dy in rotated)
        min_dy = min(dy for dx, dy in rotated)
        normalized = [(dx - min_dx, dy - min_dy) for dx, dy in rotated]
        max_dx = max(dx for dx, dy in normalized)
        max_dy = max(dy for dx, dy in normalized)
        
        for tx in range(4 - max_dx):
            for ty in range(4 - max_dy):
                coords = [(tx + dx, ty + dy) for dx, dy in normalized]
                placements.add(frozenset(coords))
    
    return list(placements)

def main():
    poly1 = read_poly()
    poly2 = read_poly()
    poly3 = read_poly()
    
    total = len(poly1) + len(poly2) + len(poly3)
    if total != 16:
        print("No")
        return
    
    poly1_placements = generate_placements(poly1)
    poly2_placements = generate_placements(poly2)
    poly3_placements = generate_placements(poly3)
    poly3_set = set(poly3_placements)
    
    all_cells = frozenset((x, y) for x in range(4) for y in range(4))
    
    for p1 in poly1_placements:
        for p2 in poly2_placements:
            if p1.isdisjoint(p2):
                combined = p1 | p2
                needed = all_cells - combined
                if needed in poly3_set:
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()