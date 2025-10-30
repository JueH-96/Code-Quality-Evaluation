def solve():
    polys = []
    for _ in range(3):
        poly = []
        for _ in range(4):
            poly.append(input())
        polys.append(poly)

    def rotate(poly):
        new_poly = [""] * 4
        for i in range(4):
            for j in range(4):
                new_poly[j] += poly[3-i][j]
        return new_poly

    def get_blocks(poly):
        blocks = []
        for r in range(4):
            for c in range(4):
                if poly[r][c] == '#':
                    blocks.append((r, c))
        return blocks

    def place(grid, poly, r_offset, c_offset, poly_num):
        new_grid = [row[:] for row in grid]
        blocks = get_blocks(poly)
        for r, c in blocks:
            new_r = r + r_offset
            new_c = c + c_offset
            if 0 <= new_r < 4 and 0 <= new_c < 4:
                if new_grid[new_r][new_c] == 0:
                    new_grid[new_r][new_c] = poly_num
                else:
                    return None
            else:
                return None
        return new_grid

    def unplace(grid, poly, r_offset, c_offset, poly_num):
        blocks = get_blocks(poly)
        for r, c in blocks:
            new_r = r + r_offset
            new_c = c + c_offset
            if 0 <= new_r < 4 and 0 <= new_c < 4:
                if grid[new_r][new_c] == poly_num:
                    grid[new_r][new_c] = 0

    def count_blocks(poly):
        count = 0
        for r in range(4):
            for c in range(4):
                if poly[r][c] == '#':
                    count += 1
        return count
    
    counts = [count_blocks(poly) for poly in polys]
    if sum(counts) != 16:
        print("No")
        return

    def solve_recursive(grid, poly_index):
        if poly_index == 3:
            for r in range(4):
                for c in range(4):
                    if grid[r][c] == 0:
                        return False
            return True

        poly = polys[poly_index]
        for _ in range(4):
            blocks = get_blocks(poly)
            min_r = min(r for r, c in blocks)
            min_c = min(c for r, c in blocks)
            for r_offset in range(4):
                for c_offset in range(4):
                    new_grid = place(grid, poly, r_offset - min_r, c_offset - min_c, poly_index + 1)
                    if new_grid:
                        if solve_recursive(new_grid, poly_index + 1):
                            return True
                        unplace(grid, poly, r_offset - min_r, c_offset - min_c, poly_index + 1)
            poly = rotate(poly)
        return False

    grid = [[0] * 4 for _ in range(4)]
    if solve_recursive(grid, 0):
        print("Yes")
    else:
        print("No")

solve()