import sys

# Memoization dictionary for g function (is_mountain)
memo_g = {}

def g(crease_idx):
    """
    Determines if the crease_idx-th crease is a mountain fold.
    Returns 1 if mountain, 0 if valley.
    crease_idx is 1-indexed.
    """
    # Problem constraints: i >= 1, A_k >= 0. So cur_i + A_k >= 1.
    # Thus, crease_idx will always be positive.
    
    if crease_idx in memo_g:
        return memo_g[crease_idx]
    
    # v2 is the exponent of 2 in the prime factorization of crease_idx.
    # (crease_idx & -crease_idx) gives the lowest set bit (which is 2^v2).
    # (2^v2).bit_length() - 1 gives v2.
    lowest_set_bit = crease_idx & -crease_idx # This is 2**v2
    v2 = lowest_set_bit.bit_length() - 1
    
    # m is the odd part of crease_idx: m = crease_idx / (2^v2)
    m = crease_idx >> v2 # Equivalent to integer division by 2**v2
    
    # A crease is a mountain fold if m % 4 == 3. Otherwise, it's a valley fold (m % 4 == 1).
    is_mountain = 1 if (m % 4 == 3) else 0
    
    memo_g[crease_idx] = is_mountain
    return is_mountain

def solve():
    N = int(sys.stdin.readline())
    A_arr = list(map(int, sys.stdin.readline().split()))

    candidate_L_values = set()
    candidate_L_values.add(0) # Base case for L

    for k_idx in range(N): 
        val_Ak = A_arr[k_idx]
        if val_Ak == 0: # This is A_1 = 0. p_k logic based on A_k!=0.
            continue

        lowest_set_bit_Ak = val_Ak & -val_Ak
        pk = lowest_set_bit_Ak.bit_length() - 1
        
        for c_offset in [-2, -1, 0, 1, 2]:
            if pk + c_offset >= 0: # L must be non-negative
                candidate_L_values.add(pk + c_offset)
    
    max_f_val = 0
    if N == 0:
        print(0)
        return

    sorted_L_values = sorted(list(candidate_L_values))
    
    for L in sorted_L_values:
        for I_multiplier in [1, 3]: 
            cur_i = I_multiplier << L 
            
            current_f_sum = 0
            for k_idx_val_A in range(N): # Iterate 0 to N-1 for A_arr
                crease_val = cur_i + A_arr[k_idx_val_A]
                current_f_sum += g(crease_val)
            
            if current_f_sum > max_f_val:
                max_f_val = current_f_sum
            
            if max_f_val == N: # Optimization
                print(N)
                return
    
    print(max_f_val)

solve()