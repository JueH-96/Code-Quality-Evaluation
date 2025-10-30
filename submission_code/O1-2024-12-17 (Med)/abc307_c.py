def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse input
    idx = 0
    H_A, W_A = map(int, input_data[idx:idx+2])
    idx += 2
    A = input_data[idx:idx+H_A]
    idx += H_A
    
    H_B, W_B = map(int, input_data[idx:idx+2])
    idx += 2
    B = input_data[idx:idx+H_B]
    idx += H_B
    
    H_X, W_X = map(int, input_data[idx:idx+2])
    idx += 2
    X = input_data[idx:idx+H_X]
    idx += H_X
    
    # Helper function to get set of black squares (row,col) for a given sheet
    def get_black_squares(sheet, H, W):
        s = set()
        for r in range(H):
            for c in range(W):
                if sheet[r][c] == '#':
                    s.add((r, c))
        return s
    
    # Get black squares for A, B, X
    blackA = get_black_squares(A, H_A, W_A)
    blackB = get_black_squares(B, H_B, W_B)
    blackX = get_black_squares(X, H_X, W_X)
    
    # If there's no black squares in any, the problem statement says each has at least one '#',
    # so we don't need an early check for empties.

    # Get bounding box for black squares in A, B
    def bounding_box(blacks):
        rvals = [r for (r,c) in blacks]
        cvals = [c for (r,c) in blacks]
        return min(rvals), max(rvals), min(cvals), max(cvals)
    
    minAr, maxAr, minAc, maxAc = bounding_box(blackA)
    minBr, maxBr, minBc, maxBc = bounding_box(blackB)
    # bounding box for X is not strictly needed to iterate over, but let's keep for reference
    # minXr, maxXr, minXc, maxXc = bounding_box(blackX)
    
    # We want to place A so that all black squares of A run within [0..H_X-1] x [0..W_X-1].
    # That implies constraints for da_r, da_c:
    #   0 <= (r + da_r) <= H_X - 1 for all r in [minAr..maxAr]
    #   0 <= (c + da_c) <= W_X - 1 for all c in [minAc..maxAc]
    # So da_r in [-minAr..(H_X - 1 - maxAr)], same for columns.
    def feasible_range(minr, maxr, H):
        # 0 <= r + d <= H - 1
        # => -minr <= d <= H - 1 - maxr
        return range(-minr, H - maxr)
    # do similarly for columns
    da_r_candidates = feasible_range(minAr, maxAr, H_X)
    da_c_candidates = feasible_range(minAc, maxAc, W_X)
    db_r_candidates = feasible_range(minBr, maxBr, H_X)
    db_c_candidates = feasible_range(minBc, maxBc, W_X)
    
    # Now brute force over these feasible ranges
    blackA_list = list(blackA)
    blackB_list = list(blackB)
    blackX_list = list(blackX)  # might not be strictly needed as a list
    
    blackX_len = len(blackX)
    
    # Convert blackX to a frozenset for quick equality check
    blackX_fset = frozenset(blackX)
    
    # We will check if union of shifted A and B is exactly blackX
    # Also, we already ensure that all squares are in range by the feasible ranges
    # so we only need to check union == blackX at the end.
    
    # To speed up repeated checks, we might do sets only once inside the loops.
    # That is straightforward:
    
    for da_r in da_r_candidates:
        for da_c in da_c_candidates:
            shiftedA = set((r + da_r, c + da_c) for (r, c) in blackA_list)
            for db_r in db_r_candidates:
                for db_c in db_c_candidates:
                    shiftedB = set((r + db_r, c + db_c) for (r, c) in blackB_list)
                    # Union
                    union_set = shiftedA.union(shiftedB)
                    # Quick length check first
                    if len(union_set) == blackX_len:
                        # Now check if exactly blackX
                        # but we must check it is exactly the set of black squares for X,
                        # i.e. positions from [0..H_X-1]x[0..W_X-1].
                        # union_set must match blackX as sets of positions within that rectangle.
                        if frozenset(union_set) == blackX_fset:
                            print("Yes")
                            return
    # If no arrangement succeeded
    print("No")

# Do not forget to call main()
main()