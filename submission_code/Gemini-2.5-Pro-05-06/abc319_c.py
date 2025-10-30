import itertools
import math

def solve():
    # Read the 3x3 grid
    C_grid = []
    for _ in range(3):
        C_grid.append(list(map(int, input().split())))

    # Define all cell coordinates (0-indexed)
    all_cells_coords = []
    for r_idx in range(3):
        for c_idx in range(3):
            all_cells_coords.append((r_idx, c_idx))

    # Define all 8 lines in the grid
    # Each line is a list of 3 (row, col) tuples
    lines = []
    # Rows
    for r_idx in range(3):
        lines.append([(r_idx, 0), (r_idx, 1), (r_idx, 2)])
    # Columns
    for c_idx in range(3):
        lines.append([(0, c_idx), (1, c_idx), (2, c_idx)])
    # Diagonals
    lines.append([(0, 0), (1, 1), (2, 2)]) # Main diagonal
    lines.append([(0, 2), (1, 1), (2, 0)]) # Anti-diagonal

    good_perm_count = 0
    total_perm_count = 0 

    # Iterate through all possible permutations of cell coordinates
    for p in itertools.permutations(all_cells_coords):
        total_perm_count += 1 

        # Create a map for efficient lookup of a cell's position (index) in permutation 'p'
        cell_to_order_idx = {cell_coord: i for i, cell_coord in enumerate(p)}

        current_permutation_is_good = True 
        
        for line_definition in lines:
            # Sort cells in this line by their appearance order in permutation 'p'
            ordered_cells_in_line = sorted(line_definition, key=lambda cell_coord: cell_to_order_idx[cell_coord])
            
            s1_coord = ordered_cells_in_line[0] 
            s2_coord = ordered_cells_in_line[1] 

            val_s1 = C_grid[s1_coord[0]][s1_coord[1]]
            val_s2 = C_grid[s2_coord[0]][s2_coord[1]]

            if val_s1 == val_s2:
                # Disappointment condition met.
                # (Guaranteed that 3rd cell in this line will have a different value)
                current_permutation_is_good = False
                break 
        
        if current_permutation_is_good:
            good_perm_count += 1
            
    # Calculate probability. Denominator could also be math.factorial(9).
    probability = good_perm_count / total_perm_count 
    
    print("{:.20f}".format(probability))

solve()