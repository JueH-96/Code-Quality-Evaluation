def rotate_90_clockwise(grid):
    return [''.join(col) for col in zip(*grid[::-1])]

def generate_placements(grid_lines):
    rotations = [grid_lines]
    grid = grid_lines
    for _ in range(3):
        grid = rotate_90_clockwise(grid)
        rotations.append(grid)
    
    placements = []
    for mat in rotations:
        points = set()
        for r in range(4):
            for c in range(4):
                if mat[r][c] == '#':
                    points.add((r, c))
                    
        if not points:
            continue
            
        min_r = min(r for r, c in points)
        max_r = max(r for r, c in points)
        min_c = min(c for r, c in points)
        max_c = max(c for r, c in points)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        
        if height > 4 or width > 4:
            continue
            
        normalized_points = set()
        for (r, c) in points:
            normalized_points.add((r - min_r, c - min_c))
            
        for i in range(0, 4 - height + 1):
            for j in range(0, 4 - width + 1):
                new_set = set()
                for (r, c) in normalized_points:
                    new_r = r + i
                    new_c = c + j
                    new_set.add((new_r, new_c))
                placements.append(new_set)
                
    return placements

def main():
    data = []
    for _ in range(12):
        data.append(input().strip())
        
    piece0_lines = data[0:4]
    piece1_lines = data[4:8]
    piece2_lines = data[8:12]
    
    s0 = sum(row.count('#') for row in piece0_lines)
    s1 = sum(row.count('#') for row in piece1_lines)
    s2 = sum(row.count('#') for row in piece2_lines)
    
    if s0 + s1 + s2 != 16:
        print("No")
        return
        
    all_grid = set((i, j) for i in range(4) for j in range(4))
    
    placements0 = generate_placements(piece0_lines)
    placements1 = generate_placements(piece1_lines)
    placements2 = generate_placements(piece2_lines)
    placements_set2 = set(frozenset(pl) for pl in placements2)
    
    for A in placements0:
        if len(A) != s0:
            continue
        for B in placements1:
            if len(B) != s1:
                continue
            if A & B:
                continue
            remaining = all_grid - (A | B)
            if len(remaining) != s2:
                continue
            if frozenset(remaining) in placements_set2:
                print("Yes")
                return
                
    print("No")

if __name__ == "__main__":
    main()