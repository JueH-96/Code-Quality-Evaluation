def solve():
    n, h, w = map(int, input().split())
    tiles_dimensions = []
    for _ in range(n):
        tiles_dimensions.append(tuple(map(int, input().split())))
    
    possible_tiles = []
    for i in range(n):
        possible_tiles.append(tiles_dimensions[i])
        if tiles_dimensions[i][0] != tiles_dimensions[i][1]:
            possible_tiles.append((tiles_dimensions[i][1], tiles_dimensions[i][0]))
            
    from itertools import combinations
    
    indices = list(range(n))
    for r in range(1, n + 1):
        for tile_indices in combinations(indices, r):
            original_tile_set = [tiles_dimensions[i] for i in tile_indices]
            orientations_count = [2 if dim[0] != dim[1] else 1 for dim in original_tile_set]
            num_orientations_combinations = 1
            for count in orientations_count:
                num_orientations_combinations *= count
                
            for orientation_index in range(num_orientations_combinations):
                current_tiles = []
                temp_orientation_index = orientation_index
                for i in range(len(original_tile_set)):
                    tile_dim = original_tile_set[i]
                    if tile_dim[0] == tile_dim[1]:
                        current_tiles.append(tile_dim)
                    else:
                        if temp_orientation_index % 2 == 0:
                            current_tiles.append(tile_dim)
                        else:
                            current_tiles.append((tile_dim[1], tile_dim[0]))
                        temp_orientation_index //= 2
                        
                total_area = sum(dim[0] * dim[1] for dim in current_tiles)
                if total_area != h * w:
                    continue
                    
                valid_dimensions = True
                for dim in current_tiles:
                    if dim[0] > h or dim[1] > w:
                        valid_dimensions = False
                        break
                if not valid_dimensions:
                    continue
                    
                grid = [[0 for _ in range(w)] for _ in range(h)]
                
                def is_valid_placement(r_start, c_start, tile_h, tile_w):
                    if r_start + tile_h > h or c_start + tile_w > w:
                        return False
                    for r in range(r_start, r_start + tile_h):
                        for c in range(c_start, c_start + tile_w):
                            if grid[r][c] == 1:
                                return False
                    return True
                    
                def place_tile(r_start, c_start, tile_h, tile_w):
                    for r in range(r_start, r_start + tile_h):
                        for c in range(c_start, c_start + tile_w):
                            grid[r][c] = 1
                            
                def remove_tile(r_start, c_start, tile_h, tile_w):
                    for r in range(r_start, r_start + tile_h):
                        for c in range(c_start, c_start + tile_w):
                            grid[r][c] = 0
                            
                def find_first_uncovered():
                    for r in range(h):
                        for c in range(w):
                            if grid[r][c] == 0:
                                return r, c
                    return None
                    
                def solve_recursive(tiles_to_place_dimensions):
                    if not tiles_to_place_dimensions:
                        if not find_first_uncovered():
                            return True
                        else:
                            return False
                            
                    first_uncovered_cell = find_first_uncovered()
                    if not first_uncovered_cell:
                        return not tiles_to_place_dimensions
                        
                    r_start, c_start = first_uncovered_cell
                    
                    for tile_index in range(len(tiles_to_place_dimensions)):
                        tile_dim = tiles_to_place_dimensions[tile_index]
                        tile_h, tile_w = tile_dim
                        
                        if is_valid_placement(r_start, c_start, tile_h, tile_w):
                            place_tile(r_start, c_start, tile_h, tile_w)
                            remaining_tiles = tiles_to_place_dimensions[:tile_index] + tiles_to_place_dimensions[tile_index+1:]
                            if solve_recursive(remaining_tiles):
                                return True
                            remove_tile(r_start, c_start, tile_h, tile_w)
                            
                    return False
                    
                if solve_recursive(current_tiles):
                    print("Yes")
                    return
                    
    print("No")

if __name__ == '__main__':
    solve()