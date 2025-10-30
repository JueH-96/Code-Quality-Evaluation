MOD = 998244353

fact = []
inv_fact = []

def nCr_mod(n, r, mod):
    global fact, inv_fact
    if r < 0 or r > n:
        return 0
    if not fact: # Check if precomputation has been done
        # This should ideally be called once globally, not per nCr_mod call
        # For competitive programming, usually called in main/solve
        raise ValueError("Factorials not precomputed")
        
    num = fact[n]
    den = (inv_fact[r] * inv_fact[n - r]) % mod
    return (num * den) % mod

def precompute_factorials_global(max_val, mod):
    global fact, inv_fact
    fact = [1] * (max_val + 1)
    inv_fact = [1] * (max_val + 1)
    for i in range(1, max_val + 1):
        fact[i] = (fact[i - 1] * i) % mod
    
    inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
    # Iterate downwards from max_val - 1 to 0
    for i in range(max_val - 1, -1, -1): 
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod

# Sum_{i=A}^B C(i, K) = C(B+1, K+1) - C(A, K+1)
# Valid for A <= B. If A > B, sum is 0.
# C(i,K) is 0 if i < K. The formula correctly handles this.
def sum_nCr_hockey_stick(B, A, K, mod):
    if A > B:
        return 0
    # K is the lower fixed index in C(i,K)
    val1 = nCr_mod(B + 1, K + 1, mod)
    val2 = nCr_mod(A, K + 1, mod) # Note: A, not A-1. C(A, K+1) is C( (A-1)+1, K+1)
    return (val1 - val2 + mod) % mod

# Calculate Sum_{x=xs..xe} Sum_{y=ys..ye} (C(x+y+2, y+1)-1)
def sum_std_dp_rect(xs, xe, ys, ye, mod):
    if xs > xe or ys > ye:
        return 0

    # Sum_{x=xs..xe} [ (C(x+ye+3, x+2) - C(x+ys+2, x+2)) - (ye-ys+1) ]
    # First part: Sum_{x=xs..xe} C(x+ye+3, x+2)
    # C(N,K) = C(N, N-K). So C(x+ye+3, x+2) = C(x+ye+3, ye+1)
    # Let M = ye+1. We sum C(x+ye+3, M) for x from xs to xe.
    # Let i = x+ye+3. When x=xs, i=xs+ye+3. When x=xe, i=xe+ye+3.
    # Sum C(i, M) for i from xs+ye+3 to xe+ye+3.
    term1_sum = sum_nCr_hockey_stick(xe + ye + 3, xs + ye + 3, ye + 1, mod)
    
    # Second part: Sum_{x=xs..xe} C(x+ys+2, x+2)
    # C(x+ys+2, x+2) = C(x+ys+2, ys)
    # Let M = ys. We sum C(x+ys+2, M) for x from xs to xe.
    # Let i = x+ys+2. When x=xs, i=xs+ys+2. When x=xe, i=xe+ys+2.
    # Sum C(i, M) for i from xs+ys+2 to xe+ys+2.
    term2_sum = sum_nCr_hockey_stick(xe + ys + 2, xs + ys + 2, ys, mod)

    total_binom_sum = (term1_sum - term2_sum + mod) % mod
    
    num_cells = ((xe - xs + 1) * (ye - ys + 1)) % mod # Total number of "-1" terms
    final_sum = (total_binom_sum - num_cells + mod) % mod
    
    return final_sum

def solve():
    W, H, L, R, D, U = map(int, input().split())

    # Max possible value for n in nCr is W+H+4 for combinatoric sums,
    # or W+H+2 for individual dp terms. Max W,H=10^6.
    # Precompute factorials up to W+H+4.
    precompute_factorials_global(W + H + 4, MOD)

    ans = sum_std_dp_rect(0, L - 1, 0, H, MOD)
    ans = (ans + sum_std_dp_rect(L, W, 0, D - 1, MOD)) % MOD
    
    dp_prev_col = [0] * (H + 1) # Stores dp[x_idx-1][y]
    if L > 0:
        for y_idx in range(D, H + 1): # Only need y >= D for region R3
            # dp[L-1][y] = C(L-1+y+2, L-1+1)-1 = C(L+y+1, L)-1
            dp_prev_col[y_idx] = (nCr_mod(L - 1 + y_idx + 2, L - 1 + 1, MOD) - 1 + MOD) % MOD
    
    # dp_curr_col_for_y_ge_D is only for y in [D,H]
    # For iterations x_idx = L to W
    for x_idx in range(L, W + 1):
        current_col_sum_for_y_ge_D_region = 0
        
        # dp_val_at_y_minus_1 stores dp[x_idx][y_idx-1]
        # For y_idx = D, this must be dp[x_idx][D-1]
        if D > 0:
            # dp[x_idx][D-1] = C(x_idx + (D-1) + 2, x_idx+1) - 1
            dp_val_at_y_minus_1 = (nCr_mod(x_idx + (D-1) + 2, x_idx+1, MOD) - 1 + MOD) % MOD
        else: # D=0
            dp_val_at_y_minus_1 = 0 # Represents dp[x_idx][-1] which is 0

        # Temporary array for dp[x_idx][y] for y in [D, H] to correctly become dp_prev_col
        # This should be outside y-loop, inside x-loop to store the full column slice
        new_dp_col_slice_for_y_ge_D = [0] * (H + 1) # Store relevant part of dp[x_idx]

        for y_idx in range(D, H + 1):
            current_dp_val = 0
            if L <= x_idx <= R and D <= y_idx <= U: # Forbidden cell
                current_dp_val = 0
            else:
                # dp[x][y] = 1 + dp[x-1][y] + dp[x][y-1]
                current_dp_val = 1
                
                # Add dp[x-1][y] term.
                # If x_idx is L, dp_prev_col has dp[L-1][y].
                # If x_idx > L, dp_prev_col has dp[x_idx-1][y] from previous iteration.
                # If L=0 and x_idx=0, then dp_prev_col is all zeros, so this term is 0.
                # This implicitly handles x_idx=0 case because dp_prev_col is initialized based on L.
                # If L=0, dp_prev_col initialized to zeros. If x_idx=0, term is zero.
                # If L>0, dp_prev_col for x_idx=L is correctly values for dp[L-1][y]
                current_dp_val = (current_dp_val + dp_prev_col[y_idx]) % MOD
                
                # Add dp[x][y-1] term. (dp_val_at_y_minus_1)
                current_dp_val = (current_dp_val + dp_val_at_y_minus_1) % MOD
            
            new_dp_col_slice_for_y_ge_D[y_idx] = current_dp_val
            current_col_sum_for_y_ge_D_region = (current_col_sum_for_y_ge_D_region + current_dp_val) % MOD
            dp_val_at_y_minus_1 = current_dp_val # Update for next y_idx in this column

        ans = (ans + current_col_sum_for_y_ge_D_region) % MOD
        dp_prev_col = new_dp_col_slice_for_y_ge_D # Current column becomes previous for next iteration

    print(ans)

solve()