def solve():
    polys = []
    for _ in range(3):
        poly = []
        for _ in range(4):
            poly.append(input())
        polys.append(poly)

    def get_blocks(poly):
        blocks = []
        for r in range(4):
            for c in range(4):
                if poly[r][c] == '#':
                    blocks.append((r, c))
        return blocks

    def rotate(blocks):
        rotated_blocks = []
        for r, c in blocks:
            rotated_blocks.append((c, 3 - r))
        min_r = min(r for r, _ in rotated_blocks)
        min_c = min(c for _, c in rotated_blocks)
        normalized_blocks = [(r - min_r, c - min_c) for r, c in rotated_blocks]
        return normalized_blocks

    def normalize(blocks):
        min_r = min(r for r, _ in blocks)
        min_c = min(c for _, c in blocks)
        return [(r - min_r, c - min_c) for r, c in blocks]

    def get_all_rotations(poly):
        blocks = get_blocks(poly)
        rotations = set()
        for _ in range(4):
            blocks = normalize(blocks)
            rotations.add(tuple(sorted(blocks)))
            blocks = rotate(blocks)
        return list(rotations)
    
    all_rotations = [get_all_rotations(poly) for poly in polys]

    def is_valid(grid):
        count = 0
        for r in range(4):
            for c in range(4):
                if grid[r][c] != 0:
                    count += 1
        return count == 16

    def place_poly(grid, poly_blocks, r_offset, c_offset, poly_id):
        new_grid = [row[:] for row in grid]
        for r, c in poly_blocks:
            nr, nc = r + r_offset, c + c_offset
            if 0 <= nr < 4 and 0 <= nc < 4 and new_grid[nr][nc] == 0:
                new_grid[nr][nc] = poly_id
            else:
                return None
        return new_grid

    def solve_recursive(grid, poly_index):
        if poly_index == 3:
            return is_valid(grid)
        
        for rotation in all_rotations[poly_index]:
            for r_offset in range(4):
                for c_offset in range(4):
                    new_grid = place_poly(grid, rotation, r_offset, c_offset, poly_index + 1)
                    if new_grid:
                        if solve_recursive(new_grid, poly_index + 1):
                            return True
        return False

    initial_grid = [[0] * 4 for _ in range(4)]
    if solve_recursive(initial_grid, 0):
        print("Yes")
    else:
        print("No")

solve()