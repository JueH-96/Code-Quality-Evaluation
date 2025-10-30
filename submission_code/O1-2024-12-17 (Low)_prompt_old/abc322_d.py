def solve():
    import sys

    # Read the input for the three polyominoes (each is a 4x4 block of characters)
    raw_data = [sys.stdin.readline().rstrip('
') for _ in range(12)]
    
    # Parse the input into three separate 4x4 grids
    grids = []
    for i in range(0, 12, 4):
        grids.append(raw_data[i:i+4])
    
    # For each grid, extract the list of occupied squares
    polyominoes = []
    for g in grids:
        coords = []
        for r in range(4):
            for c in range(4):
                if g[r][c] == '#':
                    coords.append((r, c))
        polyominoes.append(coords)
    
    # Quick check: if total number of '#' squares != 16 => No
    total_squares = sum(len(p) for p in polyominoes)
    if total_squares != 16:
        print("No")
        return
    
    # Function to rotate a set of coordinates 90 degrees clockwise around (0,0)
    def rotate_90_ccw(coords):
        # A 90° clockwise rotation about (0,0) can be turned into
        # (r, c) -> (c, -r), but we want 90° *counter* clockwise? 
        # We only need any consistent method as long as we do it 4 times to get all rotations.
        # Let's do 90° *clockwise* instead: (r, c) -> (c, -r).
        # But it doesn't matter as we're generating all 4 anyway. We'll pick one approach, say:
        #  (r, c) -> (c, -r)
        # Then we will re-normalize.
        return [(c, -r) for (r, c) in coords]
    
    # Normalize a set of coordinates so that min row/col = 0
    def normalize(coords):
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        shifted = [(r - min_r, c - min_c) for (r, c) in coords]
        return sorted(shifted)
    
    # Generate all distinct rotations (0,90,180,270) for each shape (no flips)
    # We'll store each orientation as a sorted list of (r,c) after normalization
    def all_orientations(coords):
        res = set()
        current = coords
        for _ in range(4):
            current = rotate_90_ccw(current)
            norm = tuple(normalize(current))
            res.add(norm)
        return list(res)

    # For each polyomino, gather all possible placements (within 4x4) for each orientation
    # We'll store each placement as a frozenset of (r,c) in the 4x4 board
    all_placements = []
    for poly in polyominoes:
        # Start with normalized base shape
        base = normalize(poly)
        # Generate all orientations
        orientations = all_orientations(base)
        
        placements_for_this_shape = []
        for orient in orientations:
            # orient is a tuple of (r,c), normalized so min r,c=0
            # Find shape's bounding box
            max_r = max(r for r, _ in orient)
            max_c = max(c for _, c in orient)
            # We can shift the shape so that it stays in [0..3] range
            # If shape bounding box is H x W, row shift range = 4 - H - 1, col shift range = 4 - W - 1
            # But let's keep it simpler:
            for shift_r in range(4 - max_r):
                for shift_c in range(4 - max_c):
                    placed = [(r + shift_r, c + shift_c) for (r, c) in orient]
                    # All must be in the 4x4 board => by construction we are guaranteed that now
                    placements_for_this_shape.append(frozenset(placed))
        
        # Remove duplicates (in case shape's rotations lead to the same set)
        placements_for_this_shape = list(set(placements_for_this_shape))
        all_placements.append(placements_for_this_shape)
    
    # Now we try all combinations of picking one placement from each shape
    # Check if they fill the 4x4 exactly (16 squares) and are non-overlapping
    # Because sum_of_squares is 16, we just check if union is size 16 and no overlaps.
    shape1_placements = all_placements[0]
    shape2_placements = all_placements[1]
    shape3_placements = all_placements[2]
    
    for p1 in shape1_placements:
        len_p1 = len(p1)
        for p2 in shape2_placements:
            # quick overlap check
            if p1 & p2:
                continue
            len_p2 = len(p2)
            union_p1_p2 = p1 | p2
            combined_len = len_p1 + len_p2
            for p3 in shape3_placements:
                # check if disjoint from union of p1 & p2
                if union_p1_p2 & p3:
                    continue
                if combined_len + len(p3) == 16:
                    print("Yes")
                    return
    
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()