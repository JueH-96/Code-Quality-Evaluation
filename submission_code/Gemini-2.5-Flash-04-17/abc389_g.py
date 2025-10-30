import sys

sys.setrecursionlimit(3000) # Increase recursion depth

MOD = 0

fact = []
invFact = []

def power(a, b):
    res = 1
    a %= MOD
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % MOD
        a = (a * a) % MOD
        b //= 2
    return res

def modInverse(n):
    return power(n, MOD - 2)

def precompute_factorials(max_n):
    global fact, invFact
    fact = [1] * (max_n + 1)
    invFact = [1] * (max_n + 1)
    for i in range(2, max_n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    invFact[max_n] = modInverse(fact[max_n])
    for i in range(max_n - 1, 1, -1):
        invFact[i] = (invFact[i + 1] * (i + 1)) % MOD

def nCr_mod_p(n, r):
    if r < 0 or r > n:
        return 0
    return (fact[n] * invFact[r] % MOD * invFact[n - r] % MOD)

valid_sequences = []
N_val = 0

def generate_sequences(layer_idx, current_sum, current_even_sum, current_seq):
    if current_sum == N_val:
        # A valid sequence must have s_0=1, s_k >= 1, sum s_k = N, sum even s_k = N/2
        # The way generate_sequences is called ensures s_0=1, current_sum=N_val, current_even_sum = sum even s_k
        # Also s_k >= 1 is enforced by the loop range.
        valid_sequences.append(list(current_seq))
        return
    if current_sum > N_val:
        return

    # Layer 0 must have size 1
    if layer_idx == 0:
        generate_sequences(1, 1, 1, [1])
        return

    # Subsequent layers
    # s_k must be at least 1
    for s_k in range(1, N_val - current_sum + 1):
        new_sum = current_sum + s_k
        new_even_sum = current_even_sum + (s_k if layer_idx % 2 == 0 else 0)
        generate_sequences(layer_idx + 1, new_sum, new_even_sum, current_seq + [s_k])


def solve():
    global N_val, MOD
    N_val, MOD = map(int, sys.stdin.readline().split())

    max_edges = N_val * (N_val - 1) // 2
    precompute_factorials(max_edges)

    generate_sequences(0, 0, 0, [])

    ans = [0] * (max_edges + 1) 

    for seq in valid_sequences:
        s = seq
        D = len(s) - 1 # max layer index

        # Calculate vertex assignment ways
        denom = 1
        for k in range(1, D + 1): # s_1 to s_D
            denom = (denom * fact[s[k]]) % MOD
        vertex_assignments = (fact[N_val - 1] * modInverse(denom)) % MOD
        
        # Calculate total allowed edges for this layer structure
        E_allowed_count = 0
        for k in range(D + 1): # Edges within V_k
            E_allowed_count = (E_allowed_count + nCr_mod_p(s[k], 2)) % MOD
        for k in range(D): # Edges between V_k and V_{k+1}
            E_allowed_count = (E_allowed_count + s[k] * s[k+1]) % MOD
            E_allowed_count %= MOD

        # Inclusion-Exclusion DP
        # dp_jk_sum[k] maps sum_s_im1_ji to {0: sum_even_parity, 1: sum_odd_parity}
        # This is sum of \prod_{i=1}^k \binom{s_i}{j_i} where \sum_{i=1}^k s_{i-1} j_i = sum_s_im1_ji
        # and (\sum_{i=1}^k j_i) % 2 = parity_sum_j
        
        # Base case: Before processing layers 1 to D (i.e., k=0). Empty sum.
        # sum_s_im1_ji = 0, sum_j_i = 0. Product of empty set = 1.
        # Layers indexed 0 to D. We process layers 1 to D.
        # The product/sum of j_i is over i = 1 to k.
        dp_jk_sum = {0: {0: {0: 1, 1: 0}}} # {num_layers_processed_for_j: {sum_s_im1_ji: {parity_sum_j: value}}}
        # Number of layers processed for j is from 1 to k in the loop below.
        # So dp_jk_sum[k] will correspond to sum over j_1 .. j_k
        
        
        for k in range(1, D + 1): # k is the index of the layer V_k
            next_dp_jk_sum = {}
            s_km1 = s[k-1] # Size of layer V_{k-1}
            s_k = s[k]     # Size of layer V_k

            for current_Sj in dp_jk_sum[k-1]:
                for parity_sum_j in [0, 1]:
                    current_val = dp_jk_sum[k-1][current_Sj][parity_sum_j]
                    if current_val == 0: continue

                    # Choose j_k vertices from V_k (0 <= j_k <= s_k)
                    for jk in range(s_k + 1):
                        next_Sj = current_Sj + s_km1 * jk
                        next_parity_sum_j = (parity_sum_j + jk) % 2
                        binom_sk_jk = nCr_mod_p(s_k, jk)
                        
                        if next_Sj not in next_dp_jk_sum:
                            next_dp_jk_sum[next_Sj] = {0: 0, 1: 0}
                        
                        term = (current_val * binom_sk_jk) % MOD
                        next_dp_jk_sum[next_Sj][next_parity_sum_j] = (next_dp_jk_sum[next_Sj][next_parity_sum_j] + term) % MOD
            dp_jk_sum[k] = next_dp_jk_sum

        # Now sum over all possible M values
        # The DP calculated sums for k=1..D. So the final results are in dp_jk_sum[D]
        final_jk_sums_with_parity = dp_jk_sum[D]
        
        # The contribution for a fixed sum Sj and parity p is (-1)^p * value
        # total_inclusion_exclusion_sum = sum ( (-1)^p * val ) over Sj, p
        # = sum_Sj ( val_even - val_odd ) over Sj
        
        for Sj in final_jk_sums_with_parity:
            val_even = final_jk_sums_with_parity[Sj][0]
            val_odd = final_jk_sums_with_parity[Sj][1]
            
            total_ie_term_for_Sj = (val_even - val_odd + MOD) % MOD

            # For each M from N-1 to max_edges
            for M in range(N_val - 1, max_edges + 1):
                num_graphs_M = nCr_mod_p(E_allowed_count - Sj, M)
                ans[M] = (ans[M] + vertex_assignments * total_ie_term_for_Sj * num_graphs_M) % MOD
                
    output_str = []
    for M in range(N_val - 1, max_edges + 1):
        output_str.append(str(ans[M]))

    print(" ".join(output_str))

solve()