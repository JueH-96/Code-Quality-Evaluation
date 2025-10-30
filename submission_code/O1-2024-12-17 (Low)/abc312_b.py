def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    grid = input_data[2:]

    # Helper to get the color of a cell: '#' for black, '.' for white
    def color(r, c):
        return grid[r][c]

    # Predefine the sets of cells (relative to the top-left corner of the 9x9 region)
    # that must be black and the offsets we'll use to find adjacency.
    must_be_black = set()
    # Top-left 3x3 block
    for r in range(3):
        for c in range(3):
            must_be_black.add((r, c))
    # Bottom-right 3x3 block
    for r in range(6, 9):
        for c in range(6, 9):
            must_be_black.add((r, c))

    # Offsets for adjacency (8 directions)
    adj_offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

    def is_valid_tak_code(top_row, left_col):
        # 1) Check must-be-black cells
        for (r, c) in must_be_black:
            if color(top_row + r, left_col + c) != '#':
                return False
        
        # 2) Build the set of must-be-white cells
        #    These are all cells that are adjacent to any must-be-black cell,
        #    excluding the black cells themselves.
        must_be_white = set()
        for (r, c) in must_be_black:
            for (dr, dc) in adj_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    if (nr, nc) not in must_be_black:
                        must_be_white.add((nr, nc))

        # Check all must-be-white cells are indeed white
        for (r, c) in must_be_white:
            if color(top_row + r, left_col + c) != '.':
                return False

        # If all checks passed, it's a valid Tak Code
        return True

        # Note: We do not need to enforce that must_be_white has size 14;
        # the problem's own geometry guarantees it if the pattern is correct.

    # We will check all possible 9x9 subgrids
    answers = []
    for i in range(N - 8):
        for j in range(M - 8):
            if is_valid_tak_code(i, j):
                # Output row and column in 1-based indexing
                answers.append((i + 1, j + 1))

    # Print results
    for r, c in answers:
        print(r, c)

# Don't forget to call main()
if __name__ == "__main__":
    main()