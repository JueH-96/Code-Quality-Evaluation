import sys

def solve():
    # Read H_A, W_A and sheet A
    H_A, W_A = map(int, sys.stdin.readline().split())
    A_str_list = [sys.stdin.readline().strip() for _ in range(H_A)]
    
    # Read H_B, W_B and sheet B
    H_B, W_B = map(int, sys.stdin.readline().split())
    B_str_list = [sys.stdin.readline().strip() for _ in range(H_B)]
    
    # Read H_X, W_X and sheet X
    H_X, W_X = map(int, sys.stdin.readline().split())
    X_str_list = [sys.stdin.readline().strip() for _ in range(H_X)]

    # --- Preprocessing ---
    # Store coordinates of black squares for sheet A
    black_A_coords = []
    for r in range(H_A):
        for c in range(W_A):
            if A_str_list[r][c] == '#':
                black_A_coords.append((r, c))

    # Store coordinates of black squares for sheet B
    black_B_coords = []
    for r in range(H_B):
        for c in range(W_B):
            if B_str_list[r][c] == '#':
                black_B_coords.append((r, c))
    
    # Store coordinates of black squares for sheet X as a set for efficient comparison
    black_X_set = set()
    for r in range(H_X):
        for c in range(W_X):
            if X_str_list[r][c] == '#':
                black_X_set.add((r,c))

    # --- Determine valid shift ranges ---
    # Problem guarantees at least one black square, so black_A_coords and black_B_coords are non-empty.
    
    min_r_in_A = min(r for r, c in black_A_coords)
    max_r_in_A = max(r for r, c in black_A_coords)
    min_c_in_A = min(c for r, c in black_A_coords)
    max_c_in_A = max(c for r, c in black_A_coords)
    
    min_drA = -min_r_in_A
    max_drA = H_X - 1 - max_r_in_A
    min_dcA = -min_c_in_A
    max_dcA = W_X - 1 - max_c_in_A
    
    min_r_in_B = min(r for r, c in black_B_coords)
    max_r_in_B = max(r for r, c in black_B_coords)
    min_c_in_B = min(c for r, c in black_B_coords)
    max_c_in_B = max(c for r, c in black_B_coords)

    min_drB = -min_r_in_B
    max_drB = H_X - 1 - max_r_in_B
    min_dcB = -min_c_in_B
    max_dcB = W_X - 1 - max_c_in_B

    # --- Iterate over all valid shifts for A and B ---
    for drA in range(min_drA, max_drA + 1):
        for dcA in range(min_dcA, max_dcA + 1):
            for drB in range(min_drB, max_drB + 1):
                for dcB in range(min_dcB, max_dcB + 1):
                    
                    current_formed_X_black_cells = set()
                    
                    for r_offset_A, c_offset_A in black_A_coords:
                        xr, xc = drA + r_offset_A, dcA + c_offset_A
                        current_formed_X_black_cells.add((xr, xc))
                    
                    for r_offset_B, c_offset_B in black_B_coords:
                        xr, xc = drB + r_offset_B, dcB + c_offset_B
                        current_formed_X_black_cells.add((xr, xc))
                    
                    if current_formed_X_black_cells == black_X_set:
                        print("Yes")
                        return
                        
    print("No")

solve()