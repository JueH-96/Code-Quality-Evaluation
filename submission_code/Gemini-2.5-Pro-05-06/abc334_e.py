import collections
import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    grid_chars = [sys.stdin.readline().strip() for _ in range(H)]

    MOD = 998244353

    # --- Step 1: Count red cells N_R ---
    N_R = 0
    for r in range(H):
        for c in range(W):
            if grid_chars[r][c] == '.':
                N_R += 1
    
    # Constraints: "There is at least one (i,j) such that S_{i,j} = ."
    # This means N_R >= 1.

    # --- Step 2: Calculate K_orig and component IDs for green cells ---
    comp_id_grid = [[0 for _ in range(W)] for _ in range(H)]
    visited = [[False for _ in range(W)] for _ in range(H)]
    K_orig = 0  # Number of original green connected components
    
    q = collections.deque()
    
    # Directions for neighbors (up, down, left, right)
    dr_moves = [-1, 1, 0, 0]
    dc_moves = [0, 0, -1, 1]

    for r_start in range(H):
        for c_start in range(W):
            if grid_chars[r_start][c_start] == '#' and not visited[r_start][c_start]:
                K_orig += 1 # Found a new component
                
                q.append((r_start, c_start))
                visited[r_start][c_start] = True
                comp_id_grid[r_start][c_start] = K_orig # Assign component ID (1-indexed)
                
                while q:
                    r, c = q.popleft()
                    for i in range(4): # Check all four neighbors
                        nr, nc = r + dr_moves[i], c + dc_moves[i]
                        
                        if 0 <= nr < H and 0 <= nc < W and \
                           grid_chars[nr][nc] == '#' and not visited[nr][nc]:
                            visited[nr][nc] = True
                            comp_id_grid[nr][nc] = K_orig # Assign same component ID
                            q.append((nr, nc))
                            
    # --- Step 3: Calculate sum of adjustment terms S_adj ---
    S_adj = 0
    neighbor_comp_ids_set = set() # Re-use set object

    for r_idx in range(H):
        for c_idx in range(W):
            if grid_chars[r_idx][c_idx] == '.': # This is a red cell
                neighbor_comp_ids_set.clear() 
                for i in range(4): 
                    nr, nc = r_idx + dr_moves[i], c_idx + dc_moves[i]
                    
                    if 0 <= nr < H and 0 <= nc < W and \
                       grid_chars[nr][nc] == '#': # Green neighbor
                        neighbor_comp_ids_set.add(comp_id_grid[nr][nc])
                
                k_val = len(neighbor_comp_ids_set)
                
                if k_val == 0:
                    S_adj += 1
                elif k_val >= 2:
                    S_adj += (1 - k_val)
                # If k_val == 1, adjustment is 0, S_adj does not change.

    # --- Step 4: Calculate total sum of components and then the expected value ---
    # Total sum of components P = N_R * K_orig + S_adj
    # This sum P is computed using Python's arbitrary precision integers.
    # P will be >= N_R >= 1, so it's positive.
    
    term_N_R_K_orig_val = N_R * K_orig
    total_sum_of_components_val = term_N_R_K_orig_val + S_adj
    
    # P_mod = P % MOD
    total_sum_of_components_mod = total_sum_of_components_val % MOD
        
    # Modular inverse of N_R: N_R_inv = N_R^(MOD-2) mod MOD
    N_R_inv = pow(N_R, MOD - 2, MOD)
    
    expected_value = (total_sum_of_components_mod * N_R_inv) % MOD
    
    sys.stdout.write(str(expected_value) + "
")

solve()