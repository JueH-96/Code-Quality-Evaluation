def solve():
    n, h, w = map(int, input().split())
    tiles_dims = []
    for _ in range(n):
        tiles_dims.append(tuple(map(int, input().split())))
    
    grid_area = h * w
    possible_tile_indices_subsets = []
    
    for i in range(1 << n):
        current_tiles_indices = []
        current_area = 0
        for j in range(n):
            if (i >> j) & 1:
                current_tiles_indices.append(j)
                current_area += tiles_dims[j][0] * tiles_dims[j][1]
        if current_area == grid_area:
            possible_tile_indices_subsets.append(current_tiles_indices)
            
    if not possible_tile_indices_subsets:
        print("No")
        return
        
    for tile_indices in possible_tile_indices_subsets:
        tiles_to_use = [tiles_dims[i] for i in tile_indices]
        
        def is_valid_placement(current_grid, row_start, col_start, tile_h, tile_w):
            for r in range(row_start, row_start + tile_h):
                for c in range(col_start, col_start + tile_w):
                    if current_grid[r][c] == 1:
                        return False
            return True
            
        def get_next_uncovered_cell(current_grid):
            for r in range(h):
                for c in range(w):
                    if current_grid[r][c] == 0:
                        return r, c
            return None
            
        def solve_recursive(current_grid, remaining_tiles):
            if not remaining_tiles:
                if get_next_uncovered_cell(current_grid) is None:
                    return True
                else:
                    return False
                    
            next_uncovered_cell = get_next_uncovered_cell(current_grid)
            if next_uncovered_cell is None:
                return True
            
            start_row, start_col = next_uncovered_cell
            
            current_tile_dims = remaining_tiles[0]
            rest_tiles = remaining_tiles[1:]
            
            dims_options = [current_tile_dims, (current_tile_dims[1], current_tile_dims[0])]
            
            for tile_h, tile_w in dims_options:
                if tile_h <= h and tile_w <= w:
                    for r in range(start_row, h - tile_h + 1):
                        for c in range(start_col, w - tile_w + 1):
                            if r == start_row and c == start_col:
                                if is_valid_placement(current_grid, r, c, tile_h, tile_w):
                                    next_grid = [list(row) for row in current_grid]
                                    for i in range(r, r + tile_h):
                                        for j in range(c, c + tile_w):
                                            next_grid[i][j] = 1
                                    if solve_recursive(tuple(tuple(row) for row in next_grid), rest_tiles):
                                        return True
                                        
            return False
            
        initial_grid = tuple(tuple(0 for _ in range(w)) for _ in range(h))
        if solve_recursive(initial_grid, tiles_to_use):
            print("Yes")
            return
            
    print("No")

if __name__ == '__main__':
    solve()