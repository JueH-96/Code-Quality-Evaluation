def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    # Remove any extra empty lines.
    data = [line for line in data if line.strip() != ""]
    # We expect 12 lines in total (4 for each polyomino).
    if len(data) < 12:
        return

    # Read the polyominoes.
    # Each polyomino is given by 4 lines.
    polyominoes = []
    for i in range(3):
        shape = []
        for j in range(4):
            shape.append(data[i * 4 + j])
        polyominoes.append(shape)

    # For each polyomino, extract the coordinates of '#' cells.
    pieces = []
    total_area = 0
    for shape in polyominoes:
        pts = []
        for i, line in enumerate(shape):
            for j, ch in enumerate(line):
                if ch == '#':
                    pts.append((i, j))
        total_area += len(pts)
        # Normalize the points so that its top-left is at (0,0)
        if pts:
            min_i = min(p[0] for p in pts)
            min_j = min(p[1] for p in pts)
            norm = [(p[0]-min_i, p[1]-min_j) for p in pts]
        else:
            norm = []
        pieces.append(norm)
    
    # The filled board must cover 4x4=16 squares.
    if total_area != 16:
        sys.stdout.write("No")
        return

    # Helper functions.

    def normalize(shape):
        """Normalize a list of (r, c) coordinates so that the minimum row and column become 0."""
        if not shape:
            return shape
        min_r = min(r for r, _ in shape)
        min_c = min(c for _, c in shape)
        return sorted([(r - min_r, c - min_c) for r, c in shape])

    def get_dimensions(shape):
        """Return the dimensions (height, width) of the shape's bounding box."""
        if not shape:
            return (0, 0)
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        return (max_r + 1, max_c + 1)

    def rotate(shape, h, w):
        """Rotate shape 90 degrees clockwise.
           For each (r, c), new coordinate becomes (c, h-1-r).
        """
        new_shape = [(c, h - 1 - r) for (r, c) in shape]
        return normalize(new_shape)

    # For each piece, generate all unique rotations (0, 90, 180, 270) without flipping.
    pieces_transforms = []
    for piece in pieces:
        transforms = []
        norm = normalize(piece)
        dims = get_dimensions(norm)
        current = norm
        cur_dims = dims
        for _ in range(4):
            if current not in transforms:
                transforms.append(current)
            rotated = rotate(current, cur_dims[0], cur_dims[1])
            new_dims = get_dimensions(rotated)
            current = rotated
            cur_dims = new_dims
        pieces_transforms.append(transforms)

    # Set up a 4x4 board. We'll mark cells as True when occupied.
    board = [[False] * 4 for _ in range(4)]

    def can_place(piece_shape, top, left):
        """Check if piece_shape can be placed with its top-left at (top, left)
           without going out of bounds or overlapping already placed pieces.
        """
        for (r, c) in piece_shape:
            rr = top + r
            cc = left + c
            if rr < 0 or rr >= 4 or cc < 0 or cc >= 4:
                return False
            if board[rr][cc]:
                return False
        return True

    def place(piece_shape, top, left, mark):
        """Place or remove a piece on the board. mark=True to place, False to remove."""
        for (r, c) in piece_shape:
            board[top + r][left + c] = mark

    # Backtracking DFS: try to place pieces one by one.
    def dfs(piece_index):
        if piece_index == 3:
            # All pieces placed, check that board is completely filled.
            for i in range(4):
                for j in range(4):
                    if not board[i][j]:
                        return False
            return True

        for shape in pieces_transforms[piece_index]:
            height, width = get_dimensions(shape)
            for r in range(4 - height + 1):
                for c in range(4 - width + 1):
                    if can_place(shape, r, c):
                        place(shape, r, c, True)
                        if dfs(piece_index + 1):
                            return True
                        place(shape, r, c, False)
        return False

    if dfs(0):
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")


if __name__ == '__main__':
    main()