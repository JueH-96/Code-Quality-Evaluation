# YOUR CODE HERE
import sys

def solve():
    # Read input dimensions H (height) and W (width)
    H, W = map(int, sys.stdin.readline().split())
    # Read the grid into a list of strings
    S_orig = [sys.stdin.readline().strip() for _ in range(H)]

    # Define the modulus
    MOD = 998244353

    # We want the DP width (W) to be the smaller dimension to minimize state space size (3^W).
    # If H < W, transpose the grid and swap H and W.
    if H < W:
        # Create a new grid S with dimensions W x H
        S = [['' for _ in range(H)] for _ in range(W)]
        # Populate the transposed grid
        for r in range(H):
            for c in range(W):
                S[c][r] = S_orig[r][c]
        # Convert rows of the new grid back to strings
        S = ["".join(row) for row in S]
        # Swap H and W to reflect the new dimensions
        H, W = W, H 
    else:
        # If H >= W, use the original grid
        S = S_orig

    # Now, W is min(original H, original W), which is guaranteed to be at most sqrt(200) approx 14.
    # H is max(original H, original W).
    # The state space size N = 3^W.
    N = 3**W
    
    # DP state vector V. V[config_idx] stores the number of ways to fill rows up to the current row,
    # ending with configuration `config_idx` for the current row.
    V = [0] * N
    
    # Precompute powers of 3 for efficient indexing based on configuration tuples.
    pow3 = [1] * (W + 1)
    for k in range(1, W + 1):
        pow3[k] = pow3[k-1] * 3

    # Cache for decoding config indices to tuples. Avoids recomputing.
    decode_cache = {} 
    # Function to decode a configuration index (0 to N-1) into a tuple of W digits (1, 2, or 3).
    # The index `config_idx` corresponds to the base-3 representation where digits are 0, 1, 2 mapped from 1, 2, 3.
    # Index C = sum_{j=0}^{W-1} (c_j-1) * 3^j maps tuple (c_0, ..., c_{W-1}) to C.
    def decode(config_idx):
        # Check cache first
        if config_idx in decode_cache:
            return decode_cache[config_idx]
        
        # Decode index to list of digits
        digits = [0] * W
        temp_idx = config_idx
        for j in range(W):
             # Calculate j-th digit (0, 1, or 2) and map back to 1, 2, or 3
             digits[j] = (temp_idx % 3) + 1
             # Move to next digit position
             temp_idx //= 3
        
        # Convert list to tuple for caching (tuples are hashable)
        res_tuple = tuple(digits)
        # Store in cache
        decode_cache[config_idx] = res_tuple
        return res_tuple

    # Cache for validity checks. Key is (config_tuple, row_index).
    validity_cache = {} 
    # Function to check if a configuration tuple is valid for a given row index r.
    # Checks two conditions:
    # 1. Matches pre-filled digits in S[r].
    # 2. Satisfies horizontal adjacency constraint (adjacent cells must have different digits).
    def check_validity(config_tuple, r):
        # Use cache key (tuple, row_index)
        cache_key = (config_tuple, r)
        if cache_key in validity_cache:
             return validity_cache[cache_key]

        # Check fixed cells constraint
        for c in range(W):
            # If cell S[r][c] has a fixed digit '1', '2', or '3'
            if S[r][c] != '?':
                # Check if it matches the digit in the configuration tuple
                if int(S[r][c]) != config_tuple[c]:
                    validity_cache[cache_key] = False # Cache result
                    return False # Invalid if mismatch
        
        # Check horizontal adjacency constraint
        for c in range(W - 1):
            # If adjacent cells have the same digit
            if config_tuple[c] == config_tuple[c+1]:
                validity_cache[cache_key] = False # Cache result
                return False # Invalid

        # If all checks pass, the configuration is valid for this row
        validity_cache[cache_key] = True # Cache result
        return True

    # --- Base Case: Row 0 ---
    # Clear validity cache before processing row 0
    validity_cache = {}
    # Iterate through all possible configurations for the first row
    for config_idx in range(N):
        config_tuple = decode(config_idx)
        # If the configuration is valid for row 0
        if check_validity(config_tuple, 0):
            # There is 1 way to achieve this configuration (starting point)
            V[config_idx] = 1
    
    # --- DP Transitions: Rows 1 to H-1 ---
    # Iterate through rows starting from the second row (index 1)
    for r in range(1, H):
        
        # Compute Y = V_{r-1} * T using a fast transform algorithm.
        # V_{r-1} is the DP state vector from the previous row.
        # T is an implicit transition matrix encoding vertical constraints:
        # T[prev_config_idx, current_config_idx] = 1 if prev_config[j] != current_config[j] for all j, else 0.
        # The fast transform efficiently computes the matrix-vector product V_{r-1} * T.
        
        current_Y = list(V) # Start with V_{r-1} (DP state from previous row)

        # The transform proceeds in W stages, one for each dimension (column).
        for k in range(W): # Iterate through dimensions/columns 0..W-1
            next_Y = [0] * N # Allocate space for the output of this stage k
            power_of_3_k = pow3[k] # Precomputed 3^k
            
            # Iterate through all configuration indices i (0 to N-1)
            for i in range(N): 
                 # Determine the k-th digit (0, 1, or 2) of index i in base 3.
                 # This corresponds to the value (1, 2, or 3) at column k in the configuration tuple.
                 digit_k = (i // power_of_3_k) % 3
                 
                 # Apply the transform rule based on the matrix K = [[0,1,1],[1,0,1],[1,1,0]].
                 # The value at Y[i] is the sum of values from the previous stage `current_Y`
                 # at indices corresponding to configurations that differ from `i`'s configuration only at column k,
                 # and have a value at column k that is different from `i`'s value at column k.
                 
                 if digit_k == 0: # Config i has value 1 at column k. Need contributions from configs with 2 or 3.
                     idx1 = i + power_of_3_k      # Index for config with value 2 at column k
                     idx2 = i + 2 * power_of_3_k  # Index for config with value 3 at column k
                     next_Y[i] = (current_Y[idx1] + current_Y[idx2]) % MOD
                 elif digit_k == 1: # Config i has value 2 at column k. Need contributions from configs with 1 or 3.
                     idx0 = i - power_of_3_k      # Index for config with value 1 at column k
                     idx2 = i + power_of_3_k      # Index for config with value 3 at column k
                     next_Y[i] = (current_Y[idx0] + current_Y[idx2]) % MOD
                 else: # digit_k == 2. Config i has value 3 at column k. Need contributions from configs with 1 or 2.
                     idx0 = i - 2 * power_of_3_k  # Index for config with value 1 at column k
                     idx1 = i - power_of_3_k      # Index for config with value 2 at column k
                     next_Y[i] = (current_Y[idx0] + current_Y[idx1]) % MOD
            
            # Update current_Y for the next stage
            current_Y = next_Y 

        # After W stages, current_Y holds the result Y = V_{r-1} * T
        Y = current_Y 

        # Compute V_r = Y * D_r. This filters the results in Y.
        # D_r is an implicit diagonal matrix where D_r[i,i]=1 if config i is valid for row r, else 0.
        # Effectively, we keep Y[config_idx] only if the configuration is valid for the current row r.
        
        # Clear validity cache for the new row r
        validity_cache = {} 
        
        # Initialize the DP state vector for the current row r
        next_V_final = [0] * N
        # Iterate through all configurations
        for config_idx in range(N):
            # Decode the index back to a tuple to check its validity
            config_tuple = decode(config_idx) 
            # If the configuration is valid for the current row r
            if check_validity(config_tuple, r):
                 # Set the DP value for this configuration using the computed value from Y
                 next_V_final[config_idx] = Y[config_idx]
            # Otherwise, it remains 0 (invalid path)
        
        # Update the main DP state vector V for the next iteration (row r+1)
        V = next_V_final 

    # --- Final Answer ---
    # After processing all H rows, the final DP vector V contains counts for valid configurations of the last row.
    # The total number of valid grid fillings is the sum of all values in the final V vector.
    total_count = sum(V) % MOD
    # Print the result
    print(total_count)

# Execute the solver function
solve()