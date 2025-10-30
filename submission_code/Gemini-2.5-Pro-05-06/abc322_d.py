import sys

# --- Input Reading ---
polys_raw_input = []
for _ in range(3):
    p_str_block = []
    for _ in range(4):
        p_str_block.append(sys.stdin.readline().strip())
    polys_raw_input.append(p_str_block)

# --- Preprocessing: Parse polyominoes, count cells, generate variants ---

total_hash_count = 0
poly_coords_initial = [[] for _ in range(3)] 

for i in range(3):
    current_poly_hashes = 0
    for r_idx in range(4):
        for c_idx in range(4):
            if polys_raw_input[i][r_idx][c_idx] == '#':
                current_poly_hashes += 1
                poly_coords_initial[i].append((r_idx, c_idx))
    
    total_hash_count += current_poly_hashes

if total_hash_count != 16:
    print("No")
    sys.exit()

def normalize_coords(coords):
    if not coords:
        return []
    min_r = min(r for r, c in coords)
    min_c = min(c for r, c in coords)
    return sorted([(r - min_r, c - min_c) for r, c in coords])

def rotate_coords(coords_normalized): # 90 deg clockwise
    if not coords_normalized:
        return []
    
    max_r_val = 0
    for r_coord, _ in coords_normalized:
        if r_coord > max_r_val:
            max_r_val = r_coord
    
    rotated_raw = []
    for r_coord, c_coord in coords_normalized:
        rotated_raw.append((c_coord, max_r_val - r_coord))
    
    return normalize_coords(rotated_raw)

all_poly_variants = [] 
for i in range(3):
    base_shape_normalized = normalize_coords(poly_coords_initial[i])
    
    current_shape_variant = base_shape_normalized
    unique_shapes_for_this_poly = set() 

    for _ in range(4): 
        unique_shapes_for_this_poly.add(tuple(current_shape_variant))
        current_shape_variant = rotate_coords(current_shape_variant)
    
    all_poly_variants.append([list(s_tuple) for s_tuple in unique_shapes_for_this_poly])

# --- Backtracking solver ---
grid = [[0]*4 for _ in range(4)]

def can_place_all_recursively(poly_idx_to_place):
    if poly_idx_to_place == 3:
        return True

    for shape_variant in all_poly_variants[poly_idx_to_place]:
        max_r_in_shape = 0
        max_c_in_shape = 0
        if shape_variant: 
            for r_coord, c_coord in shape_variant:
                if r_coord > max_r_in_shape: max_r_in_shape = r_coord
                if c_coord > max_c_in_shape: max_c_in_shape = c_coord
        
        shape_height = max_r_in_shape + 1
        shape_width = max_c_in_shape + 1

        for dr in range(4 - shape_height + 1): 
            for dc in range(4 - shape_width + 1): 
                possible_to_place_here = True
                actual_coords_on_grid = [] 

                for r_s, c_s in shape_variant: 
                    gr, gc = dr + r_s, dc + c_s 
                    
                    if grid[gr][gc] == 1: 
                        possible_to_place_here = False
                        break 
                    actual_coords_on_grid.append((gr,gc))
                
                if possible_to_place_here:
                    for gr_fill, gc_fill in actual_coords_on_grid:
                        grid[gr_fill][gc_fill] = 1
                    
                    if can_place_all_recursively(poly_idx_to_place + 1):
                        return True 
                    
                    for gr_fill, gc_fill in actual_coords_on_grid:
                        grid[gr_fill][gc_fill] = 0
                            
    return False

if can_place_all_recursively(0):
    print("Yes")
else:
    print("No")