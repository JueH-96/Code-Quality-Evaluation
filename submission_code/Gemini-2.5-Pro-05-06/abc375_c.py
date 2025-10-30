import sys

def main():
    N = int(sys.stdin.readline())
    initial_grid = [list(sys.stdin.readline().strip()) for _ in range(N)]
    
    final_grid = [['' for _ in range(N)] for _ in range(N)]
    
    for r_orig in range(N):
        for c_orig in range(N):
            char_value = initial_grid[r_orig][c_orig]
            
            # Calculate depth 'd' (0-indexed distance from the nearest border)
            # A cell (r,c) is on layer 'd' if d = min(r, c, N-1-r, N-1-c).
            # The outermost layer is d=0.
            d = min(r_orig, c_orig, N - 1 - r_orig, N - 1 - c_orig)
            
            # The value initially at (r_orig, c_orig) is part of operations
            # for i = 1, 2, ..., d+1 (1-indexed operation loop).
            # So it undergoes d+1 transformations by T.
            # Each transformation is T(r,c) = (c, N-1-r) in 0-indexed coordinates.
            num_total_transformations = d + 1
            
            # Since T^4 is the identity transformation, we only care about
            # num_total_transformations mod 4.
            effective_transformations = num_total_transformations % 4
            
            # Point whose value is being tracked is (r_orig, c_orig).
            # We need to find its final position after `effective_transformations` applications of T.
            # Let P_start = (r_orig, c_orig). We want to compute T^eff(P_start).
            
            # Initial point for transformation sequence
            pt_r, pt_c = r_orig, c_orig
            
            # Determine final coordinates (final_r, final_c)
            if effective_transformations == 0: 
                # num_total_transformations is a multiple of 4.
                # T^4 is identity, so T^(multiple of 4) is identity.
                final_r, final_c = pt_r, pt_c
            elif effective_transformations == 1: # T(pt_r, pt_c)
                final_r, final_c = pt_c, N - 1 - pt_r
            elif effective_transformations == 2: # T^2(pt_r, pt_c)
                final_r, final_c = N - 1 - pt_r, N - 1 - pt_c
            else: # effective_transformations == 3, T^3(pt_r, pt_c)
                final_r, final_c = N - 1 - pt_c, pt_r
                
            final_grid[final_r][final_c] = char_value
            
    for r_idx in range(N):
        sys.stdout.write("".join(final_grid[r_idx]) + "
")

if __name__ == '__main__':
    main()