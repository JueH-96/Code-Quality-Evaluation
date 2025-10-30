import sys

def solve():
    N, L, R = map(int, sys.stdin.readline().split())

    current_sum_mod_100 = 0
    current_L_ptr = L # Represents the start of the current unprocessed part of [L,R]

    while current_L_ptr <= R:
        # Find the largest k (best_k_val) such that:
        # 1. current_L_ptr is a multiple of 2^k
        # 2. The block [current_L_ptr, current_L_ptr + 2^k - 1] is contained in [current_L_ptr, R]
        #    (i.e., current_L_ptr + 2^k - 1 <= R)
        
        best_k_val = -1 # Will store the exponent i for the query
        
        # Iterate k_val_candidate from N down to 0.
        # (1 << k_val_candidate) is 2^k_val_candidate
        for k_val_candidate in range(N, -1, -1):
            block_size = (1 << k_val_candidate)
            if current_L_ptr % block_size == 0: # Condition 1: current_L_ptr is a multiple of block_size
                if current_L_ptr + block_size - 1 <= R: # Condition 2: block fits in remaining range
                    best_k_val = k_val_candidate
                    break # Found largest k_val_candidate, exit this inner loop
        
        # At least k_val_candidate=0 (block_size=1) will always satisfy the conditions
        # as long as current_L_ptr <= R. So, best_k_val will be >= 0.

        # Query parameters for the judge:
        # i is the power: best_k_val
        # j is the block index: current_L_ptr / (2^best_k_val)
        query_i = best_k_val
        query_j = current_L_ptr // (1 << best_k_val)
        
        print(f"? {query_i} {query_j}", flush=True)
        
        T_response = int(sys.stdin.readline())
        
        if T_response == -1:
            # An error occurred (e.g. invalid query, or too many queries).
            # Per problem spec, terminate immediately.
            sys.exit() 
            
        current_sum_mod_100 = (current_sum_mod_100 + T_response) % 100
        
        # Advance current_L_ptr to the start of the next segment
        current_L_ptr += (1 << best_k_val)
        
    # All segments covering [L,R] have been processed. Print the final sum.
    print(f"! {current_sum_mod_100}", flush=True)

# Call the solver function
solve()