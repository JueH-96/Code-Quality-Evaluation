def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return

    # There are 12 lines, 4 per polyomino.
    # Each polyomino is represented in a 4x4 grid.
    # We'll parse them into sets of coordinates for '#' cells.
    pieces = []  # This will store for each piece a list of its placement variants.
    areas = []   # Store the area (number of cells) for each piece.
    
    # Helper: normalize a shape (a collection of (r,c)) by shifting it so that minimum row and column become 0.
    def normalize(shape):
        min_r = min(r for r, c in shape)
        min_c = min(c for r, c in shape)
        normalized = tuple(sorted((r - min_r, c - min_c) for r, c in shape))
        return normalized

    # Helper: rotate a normalized shape 90 degrees clockwise.
    def rotate(shape):
        # shape is assumed normalized: its coordinates start at 0.
        # Compute current dimensions.
        h = max(r for r, c in shape) + 1
        # Rotate each cell: (r, c) -> (c, h-1 - r)
        rotated = tuple(sorted((c, h - 1 - r) for r, c in shape))
        return normalize(rotated)
    
    # Helper: generate all distinct rotations (0, 90, 180, 270 degrees) for the shape.
    def get_variants(shape):
        # shape is a normalized tuple of (r, c)
        variants = set()
        current = normalize(shape)
        for _ in range(4):
            variants.add(current)
            current = rotate(current)
        return list(variants)
    
    # There are 3 pieces, each defined by 4 lines.
    for i in range(3):
        piece_lines = data[i*4:(i+1)*4]
        shape = []
        for r in range(4):
            for c in range(4):
                if piece_lines[r][c] == '#':
                    shape.append((r, c))
        # According to constraints, each polyomino is non-empty.
        normalized_shape = normalize(shape)
        # The area remains same in all rotations.
        areas.append(len(normalized_shape))
        # Generate possible rotations (variants). No flips allowed.
        variants_raw = get_variants(normalized_shape)
        # For each variant, precompute its bounding box (height and width).
        variants = []
        for var in variants_raw:
            # var is a tuple of (r,c). Compute bounding height and width.
            h = max(r for r, c in var) + 1
            w = max(c for r, c in var) + 1
            variants.append((var, h, w))
        pieces.append(variants)
    
    # Before trying placements, check if total area equals 16 (the board size 4x4).
    if sum(areas) != 16:
        sys.stdout.write("No")
        return

    full_mask = (1 << 16) - 1  # Bitmask for a full 4x4 grid (16 cells set to 1).
    
    # We'll use backtracking. Represent the grid as a bitmask where bit (r*4+c) is 1 if cell (r,c) is occupied.
    # Initially, board_mask = 0 (all cells free).
    # For each piece, try all variants and all translations that keep the piece inside the board.
    def backtrack(piece_index, board_mask):
        if piece_index == 3:
            # All pieces placed; board must be fully covered.
            return board_mask == full_mask

        # Try each variant for the current piece.
        for variant, h, w in pieces[piece_index]:
            # For each translation that keeps the piece inside the grid.
            for r0 in range(0, 4 - h + 1):
                for c0 in range(0, 4 - w + 1):
                    candidate_mask = 0
                    # For each cell in the variant, compute its placed position.
                    for dr, dc in variant:
                        r = r0 + dr
                        c = c0 + dc
                        pos = r * 4 + c
                        candidate_mask |= (1 << pos)
                    # Check if this placement overlaps with already placed pieces.
                    if board_mask & candidate_mask:
                        continue
                    new_mask = board_mask | candidate_mask
                    if backtrack(piece_index + 1, new_mask):
                        return True
        return False

    if backtrack(0, 0):
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()