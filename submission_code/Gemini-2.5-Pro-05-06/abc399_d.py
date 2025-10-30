import sys

def solve():
    N = int(sys.stdin.readline())
    A_list = list(map(int, sys.stdin.readline().split()))

    # Store 1-indexed positions for each number 1 to N
    # positions_raw[k] will store the two 1-indexed positions of k
    positions_raw = [[] for _ in range(N + 1)] 
    for i, val_in_A in enumerate(A_list):
        positions_raw[val_in_A].append(i + 1) 

    # LR_pairs maps a candidate number k_val to its (L_k, R_k) pair of positions.
    # A number k_val is a candidate if its occurrences are not adjacent (R_k - L_k > 1).
    LR_pairs = {} 
    for k_val in range(1, N + 1):
        # Each number appears exactly twice. positions_raw[k_val] will have two elements.
        # These are appended in scan order, so pos_list[0] < pos_list[1].
        L_k, R_k = positions_raw[k_val][0], positions_raw[k_val][1] 
        
        if R_k - L_k > 1: # Check for non-adjacency
            LR_pairs[k_val] = (L_k, R_k)

    # For efficient lookup: map (L,R) position pairs back to the number k_val.
    # Only candidate (L,R) pairs are stored.
    lookup_L_R_to_k_val = {}
    for k_val, (L, R) in LR_pairs.items():
        lookup_L_R_to_k_val[(L, R)] = k_val
    
    count = 0
    
    # Iterate 'a' (represented by k1_val) from 1 to N.
    # We are looking for pairs (a,b) where a < b.
    for k1_val in range(1, N + 1): 
        if k1_val not in LR_pairs: # 'a' must be a candidate
            continue
        
        L1, R1 = LR_pairs[k1_val] # Positions for 'a'
        
        # Try to find 'b' (represented by k2_val) for Type A pattern
        # Pattern: a=(L1,R1), b=(L1+1,R1+1).
        # Condition for this pattern: R1-L1 > 1 (already true for k1_val as it's a candidate).
        # This also implies R2-L2 = (R1+1)-(L1+1) = R1-L1 > 1, so k2_val is also non-adjacent.
        target_L2_A = L1 + 1
        target_R2_A = R1 + 1
        if (target_L2_A, target_R2_A) in lookup_L_R_to_k_val:
            k2_val = lookup_L_R_to_k_val[(target_L2_A, target_R2_A)]
            if k1_val < k2_val: # Ensure a < b
                count += 1
        
        # Try to find 'b' (represented by k2_val) for Type B pattern
        # Pattern: a=(L1,R1), b=(L1+1,R1-1).
        # Condition for this pattern: R1-L1 > 3.
        # This implies R1-L1 > 1 (for k1_val).
        # And for k2_val, R2-L2 = (R1-1)-(L1+1) = R1-L1-2. If R1-L1 > 3, then R1-L1-2 > 1.
        # Also, R1-L1 > 3 => R1-L1 > 2 => L1+1 < R1-1, so L2 < R2 for b.
        if R1 - L1 > 3:
            target_L2_B = L1 + 1
            target_R2_B = R1 - 1
            if (target_L2_B, target_R2_B) in lookup_L_R_to_k_val:
                k2_val = lookup_L_R_to_k_val[(target_L2_B, target_R2_B)]
                if k1_val < k2_val: # Ensure a < b
                    count += 1
    
    sys.stdout.write(str(count) + "
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()