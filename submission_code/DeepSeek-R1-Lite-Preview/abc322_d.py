def rotate(poly):
    return [list(reversed(col)) for col in zip(*poly)]

def get_bounding_box(poly):
    min_row, max_row = 4, -1
    min_col, max_col = 4, -1
    for r in range(4):
        for c in range(4):
            if poly[r][c] == '#':
                if r < min_row:
                    min_row = r
                if r > max_row:
                    max_row = r
                if c < min_col:
                    min_col = c
                if c > max_col:
                    max_col = c
    return min_row, max_row, min_col, max_col

def get_relative_positions(poly):
    min_r, max_r, min_c, max_c = get_bounding_box(poly)
    relative_positions = []
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if poly[r][c] == '#':
                relative_positions.append((r - min_r, c - min_c))
    return relative_positions

def get_placements(poly):
    placements = []
    rotations = [poly]
    for _ in range(3):
        rotations.append(rotate(rotations[-1]))
    for rotation in rotations:
        relative_positions = get_relative_positions(rotation)
        min_r, max_r, min_c, max_c = get_bounding_box(rotation)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        for x in range(4 - height + 1):
            for y in range(4 - width + 1):
                cells = set()
                for dr, dc in relative_positions:
                    nr = x + dr
                    nc = y + dc
                    cells.add((nr, nc))
                placements.append(cells)
    return placements

def main():
    # Read input for poly1
    poly1 = [list(input().strip()) for _ in range(4)]
    # Read input for poly2
    poly2 = [list(input().strip()) for _ in range(4)]
    # Read input for poly3
    poly3 = [list(input().strip()) for _ in range(4)]
    
    # Generate all placements for each polyomino
    placements1 = get_placements(poly1)
    placements2 = get_placements(poly2)
    placements3 = get_placements(poly3)
    
    # Define the full grid
    full_grid = set((r, c) for r in range(4) for c in range(4))
    
    # Check all combinations of placements
    for p1 in placements1:
        for p2 in placements2:
            if p1.isdisjoint(p2):
                combined = p1.union(p2)
                for p3 in placements3:
                    if p3.isdisjoint(combined) and combined.union(p3) == full_grid:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()