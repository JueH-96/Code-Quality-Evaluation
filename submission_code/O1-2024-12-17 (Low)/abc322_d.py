def main():
    import sys
    input_data = sys.stdin.read().strip().split('
')
    
    # Read the three polyomino grids (each has 4 lines of 4 chars)
    poly_raw = []
    for i in range(3):
        block = input_data[4*i : 4*(i+1)]
        poly_raw.append(block)
    
    # Convert each 4x4 block to coordinates of '#'
    # We'll store as lists of (r,c) for each polyomino
    # Then we'll remove the offset so the top-left corner is (0,0)
    # because we only need relative positions to handle translations later.
    all_polys = []
    for block in poly_raw:
        coords = []
        for r in range(4):
            for c in range(4):
                if block[r][c] == '#':
                    coords.append((r, c))
        # Normalize so the top-left of bounding box is at (0,0)
        min_r = min(p[0] for p in coords)
        min_c = min(p[1] for p in coords)
        normed = [(p[0] - min_r, p[1] - min_c) for p in coords]
        all_polys.append(normed)
    
    # If total number of '#' across the three shapes is not 16, answer is automatically No
    total_squares = sum(len(x) for x in all_polys)
    if total_squares != 16:
        print("No")
        return
    
    # Function to generate all rotations (0,90,180,270) of a shape (without flipping).
    # We'll re-normalize each rotation so that top-left is at (0,0).
    def rotations(coords):
        # coords is a list of (r, c)
        # rot90: (r, c) -> (c, -r)    // Then we'll re-normalize
        # We'll generate 4 distinct sets of coords
        # and return them as lists in normalized form
        res = []
        current = coords
        for _ in range(4):
            # Normalize current
            min_r = min(p[0] for p in current)
            min_c = min(p[1] for p in current)
            norm = sorted((p[0] - min_r, p[1] - min_c) for p in current)
            res.append(norm)
            # rotate current by 90 deg (about origin)
            current = [(p[1], -p[0]) for p in current]
        # We only want unique orientations
        unique = []
        seen = set()
        for rset in res:
            rtuple = tuple(rset)
            if rtuple not in seen:
                seen.add(rtuple)
                unique.append(rset)
        return unique
    
    # Precompute all orientations for the 3 polyominoes
    poly_orients = []
    for i in range(3):
        poly_orients.append(rotations(all_polys[i]))
    
    # We'll do a backtracking approach:
    # We'll represent the 4x4 board usage with a bitmask (16 bits) or a set of used squares.
    # For convenience we'll use an integer with bits from top to bottom, left to right.
    
    # A helper to transform row,col into a bit position in [0..15]
    def to_bit(r, c):
        return (r << 2) + c  # r*4 + c
    
    # Check if we can place a shape (list of (r,c)) in orientation "coords" at top-left offset (dr,dc)
    # on an unused board state "board". Return new board bitmask if valid, or None if it doesn't fit.
    def try_place(coords, dr, dc, board):
        new_board = board
        for (r, c) in coords:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= 4 or cc < 0 or cc >= 4:  # out of bounds
                return None
            bitpos = to_bit(rr, cc)
            if (board & (1 << bitpos)) != 0:  # already occupied
                return None
            new_board |= (1 << bitpos)
        return new_board
    
    # We'll do a simple DFS in shape order 0->1->2
    # If we place all 3, check if board is fully covered
    def dfs(idx, board):
        if idx == 3:
            # check if board is full (all 16 squares used)
            return (board == (1 << 16) - 1)  # (1<<16)-1 is binary 1111111111111111 (16 bits)
        for orient in poly_orients[idx]:
            # orient is a list of (r,c)
            # bounding box might be up to 4 wide, but let's just brute force
            # we can shift from dr,dc in [0.. something], but let's be safe:
            # We can guess dr,dc in range(0..4), but to be safe let's go up to 4 because normalizing might shift.
            # We'll just try dr in [0..4], dc in [0..4], check if it fits
            # Actually we can do a safe bound: max row or col in orient can be up to 3 if it was in a 4x4 bounding box
            # We'll place it so the shape doesn't go out of 4x4
            max_r = max(p[0] for p in orient)
            max_c = max(p[1] for p in orient)
            for dr in range(4 - max_r):
                for dc in range(4 - max_c):
                    new_board = try_place(orient, dr, dc, board)
                    if new_board is not None:
                        if dfs(idx + 1, new_board):
                            return True
        return False
    
    # Start DFS with idx=0, board=0 (empty)
    if dfs(0, 0):
        print("Yes")
    else:
        print("No")

# Do not forget to call main()!
if __name__ == "__main__":
    main()