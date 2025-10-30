import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    # P_list uses 0-indexed internally, values are 1 to N.
    P_list = list(map(int, sys.stdin.readline().split())) 

    MOD = 998244353

    # Fenwick tree state and functions for I_0 calculation
    # Values in P_list are 1 to N. BIT is 1-indexed, so size N+1.
    bit_i0_tree = [0] * (N + 1)
    
    def update_i0(idx, delta):
        while idx <= N:
            bit_i0_tree[idx] += delta
            idx += idx & (-idx)
    
    def query_prefix_i0(idx):
        s = 0
        if idx < 1: return 0 # Querying for prefix sum up to 0 or negative is 0.
        while idx > 0:
            s += bit_i0_tree[idx]
            idx -= idx & (-idx)
        return s 

    # query_suffix_i0(val): count of elements currently in BIT that are >= val
    def query_suffix_i0(idx):
        if idx > N: return 0
        # Total elements in BIT - elements < idx
        return (query_prefix_i0(N) - query_prefix_i0(idx - 1))

    val_I0 = 0
    for i in range(N): # P_list[0] to P_list[N-1]
        p_val = P_list[i]
        # Number of elements already processed and > p_val
        val_I0 = (val_I0 + query_suffix_i0(p_val + 1)) % MOD
        update_i0(p_val, 1)

    if K == 1: # I_0 is the answer
        print(val_I0)
        return

    # Fenwick tree state and functions for S_in calculation
    bit_s_in_tree = [0] * (N + 1)

    def update_s_in(idx, delta):
        while idx <= N:
            bit_s_in_tree[idx] += delta
            idx += idx & (-idx)
    
    def query_prefix_s_in(idx): 
        s = 0
        if idx < 1: return 0
        while idx > 0:
            s += bit_s_in_tree[idx]
            idx -= idx & (-idx)
        return s 

    # query_suffix_s_in(val): count of elements currently in window that are >= val
    def query_suffix_s_in(idx): 
        if idx > N: return 0
        return (query_prefix_s_in(N) - query_prefix_s_in(idx - 1))

    current_inv_in_window = 0
    # Populate for first window P_list[0] ... P_list[K-1]
    for i in range(K):
        p_val = P_list[i]
        # Number of elements in P_list[0...i-1] > p_val
        current_inv_in_window = (current_inv_in_window + query_suffix_s_in(p_val + 1)) % MOD
        update_s_in(p_val, 1)
    
    total_inv_in_windows_sum = current_inv_in_window

    # Slide window: current window P_list[i]...P_list[i+K-1], next P_list[i+1]...P_list[i+K]
    # Loop i from 0 to N-K-1 (inclusive)
    # This loop runs N-K times. Window start indices are 0, 1, ..., N-K.
    for i in range(N - K): 
        p_out = P_list[i]       # Element leaving window
        p_in = P_list[i+K]    # Element entering window

        # Remove p_out:
        # It was the first element. Inversions (p_out, P_j) for j in [i+1, i+K-1] are removed.
        # Number of P_j in P_list[i+1...i+K-1] smaller than p_out must be subtracted.
        update_s_in(p_out, -1) # p_out is no longer in BIT for this calculation
                               # The elements P_list[i+1...i+K-1] are currently in BIT.
        current_inv_in_window = (current_inv_in_window - query_prefix_s_in(p_out - 1) + MOD) % MOD
        
        # Add p_in:
        # It is the last element. Inversions (P_j, p_in) for j in [i+1, i+K-1] are added.
        # Number of P_j in P_list[i+1...i+K-1] greater than p_in must be added.
        # These P_j are currently in the BIT.
        current_inv_in_window = (current_inv_in_window + query_suffix_s_in(p_in + 1)) % MOD
        update_s_in(p_in, 1) # p_in is added to BIT

        total_inv_in_windows_sum = (total_inv_in_windows_sum + current_inv_in_window) % MOD
    
    # Modular arithmetic helper functions
    def power(a, b):
        res = 1
        a %= MOD
        while b > 0:
            if b % 2 == 1:
                res = (res * a) % MOD
            a = (a * a) % MOD
            b //= 2
        return res

    def inv(n):
        return power(n, MOD - 2)

    # Calculate term_K = K(K-1)/4 mod MOD
    inv4 = inv(4)
    term_K_numerator = K * (K - 1) 
    term_K = term_K_numerator % MOD * inv4 % MOD
    
    # Calculate term_Sin = (S_in / (N-K+1)) mod MOD
    num_windows = N - K + 1
    invNMK1 = inv(num_windows)
    term_Sin = total_inv_in_windows_sum * invNMK1 % MOD
    
    ans = (val_I0 + term_K - term_Sin + MOD) % MOD
    
    print(ans)

solve()