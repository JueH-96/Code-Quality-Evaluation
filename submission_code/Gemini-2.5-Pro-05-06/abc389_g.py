import sys

def solve():
    N, P = map(int, sys.stdin.readline().split())
    MOD = P

    # Max N for C(N,K) can be up to N_E*N_O or E0+E1.
    # N_E = N//2, N_O = N//2. Max N_E, N_O is 15 for N=30.
    # N_E*N_O = 15*15 = 225.
    # E0 = C(14,2) = 91 if N_E-1 >=2 else 0.
    # E1 = C(15,2) = 105 if N_O >=2 else 0.
    # E0+E1 = 91+105 = 196.
    # So max C_N needed is around 225. Also N itself can be up to 30.
    # Max possible edges C(N,2) = C(30,2) = 435.
    
    # Max argument for C(n,k) overall.
    # For ways_to_choose_partition: N-1 (max 29)
    # For T_arr: max_edges_PE + max_edges_PO (max 196)
    # For B_arr (terms C[edges_possible_for_mb][m_b]): edges_possible_for_mb (max 225)
    # So MAX_COMB_ARG_N = 225 should be enough for C[n][k] table.
    # The loops for m_prime, m_b, etc. go up to these values too.
    # For ans_for_M, M_val goes up to C(N,2) = 435.
    
    # Max possible value of n in C(n, k)
    max_n_for_C = 0
    if N >= 2: # C(N,2) used for max_total_M_slots and loop range for M_val
      max_n_for_C = max(max_n_for_C, N*(N-1)//2) 
    
    N_E_val = N // 2
    N_O_val = N // 2
    
    max_n_for_C = max(max_n_for_C, N_E_val * N_O_val) # For edges_possible_for_mb
    
    max_edges_PE_val = 0
    if N_E_val - 1 >= 2:
        max_edges_PE_val = (N_E_val - 1) * (N_E_val - 2) // 2
    max_edges_PO_val = 0
    if N_O_val >= 2:
        max_edges_PO_val = N_O_val * (N_O_val - 1) // 2
    max_n_for_C = max(max_n_for_C, max_edges_PE_val + max_edges_PO_val) # For T_arr argument
    max_n_for_C = max(max_n_for_C, N) # For C[N-1] etc.

    C = [[0] * (max_n_for_C + 1) for _ in range(max_n_for_C + 1)]
    for i in range(max_n_for_C + 1):
        C[i][0] = 1
        if i > 0 : # C[i][j] uses C[i-1][...]
            for j in range(1, i + 1):
                C[i][j] = (C[i-1][j-1] + C[i-1][j]) % MOD
    
    ways_to_choose_partition = 0
    if N_E_val == 0: # N=0, only V_E={1} if N=0 but N>=2. Case N_E_val-1 < 0
         ways_to_choose_partition = 0 # Should not happen for N>=2
    elif N_E_val - 1 < 0 : # N=0, or N=1 (not possible N is even)
        ways_to_choose_partition = 0
    elif N-1 < N_E_val -1 : # Not enough vertices to choose from
        ways_to_choose_partition = 0
    else:
         ways_to_choose_partition = C[N-1][N_E_val - 1]


    max_m_prime = max_edges_PE_val + max_edges_PO_val
    T_arr = [0] * (max_m_prime + 1)
    if max_m_prime >=0: # If N is too small, max_m_prime can be 0
        for m_prime in range(max_m_prime + 1):
            T_arr[m_prime] = C[max_m_prime][m_prime]
    
    max_m_b = N_E_val * N_O_val
    B_arr = [0] * (max_m_b + 1)

    for m_b in range(max_m_b + 1):
        term_sum = 0
        for s0 in range(2): # 0 for not isolated, 1 for isolated for vertex 1
            if s0 == 1 and N_E_val == 0: continue # Vertex 1 must exist to be isolated
            
            # Max s1 is N_E_val-1 (size of S_E \ {1})
            # Max s2 is N_O_val (size of S_O)
            for s1 in range(N_E_val -1 + 1): # Iterate up to N_E_val-1 chosen from S_E\{1}
                for s2 in range(N_O_val + 1):
                    term_coeff_s0 = C[1][s0] # ways to choose state of vertex 1
                    term_coeff_s1 = C[N_E_val-1][s1] if N_E_val-1 >= s1 else 0
                    term_coeff_s2 = C[N_O_val][s2] if N_O_val >= s2 else 0
                    
                    term_coeff = term_coeff_s0 * term_coeff_s1 % MOD * term_coeff_s2 % MOD
                    
                    remaining_S_E = N_E_val - s0 - s1
                    remaining_S_O = N_O_val - s2
                    
                    edges_possible_for_mb = 0
                    if remaining_S_E >= 0 and remaining_S_O >= 0:
                        edges_possible_for_mb = remaining_S_E * remaining_S_O
                    
                    term_val = C[edges_possible_for_mb][m_b] if edges_possible_for_mb >= m_b else 0
                    current_term = term_coeff * term_val % MOD
                    
                    if (s0 + s1 + s2) % 2 == 1:
                        term_sum = (term_sum - current_term + MOD) % MOD
                    else:
                        term_sum = (term_sum + current_term) % MOD
        B_arr[m_b] = term_sum

    max_M_graph = 0
    if N >=2: max_M_graph = C[N][2]
    
    ans_for_M_fixed_partition = [0] * (max_M_graph + 1)
    
    # Max M value for fixed partition could be max_m_b + max_m_prime
    # For N=30, this is 225 + 196 = 421.
    # The problem asks M up to C(N,2) which is 435 for N=30.
    # This implies that "forbidden edges" might not be relevant to M's range.
    # The M is total edges in the graph.
    # However, the construction ensures we only use allowed slots.
    # So M cannot exceed max_m_b + max_m_prime.
    
    # Loop for M_val (total edges in allowed slots for a fixed partition)
    # max edges for fixed partition is max_m_b + max_m_prime
    # For N=2, N_E=1,N_O=1. max_edges_PE=0, max_edges_PO=0. max_m_prime=0. T_arr=[1].
    # max_m_b = 1*1=1. B_arr might be [0,1].
    # This loop should go up to max_m_b + max_m_prime
    current_max_M_allowed_slots = max_m_b + max_m_prime

    for M_val in range(current_max_M_allowed_slots + 1): 
        current_ans_M_val = 0
        for m_b_val in range(min(M_val, max_m_b) + 1):
            m_prime_val = M_val - m_b_val
            if 0 <= m_prime_val < len(T_arr): # Check m_prime_val is valid index for T_arr
                term = B_arr[m_b_val] * T_arr[m_prime_val] % MOD
                current_ans_M_val = (current_ans_M_val + term) % MOD
        ans_for_M_fixed_partition[M_val] = current_ans_M_val
        
    results = []
    min_M_overall = N - 1
    max_M_overall = C[N][2] if N>=2 else 0

    for M in range(min_M_overall, max_M_overall + 1):
        # If M exceeds the number of available slots for edges in our construction, count is 0.
        val_for_M = 0
        if M <= current_max_M_allowed_slots : # M is total edges, check if it's possible with allowed slots
             val_for_M = ans_for_M_fixed_partition[M]

        final_ans_for_this_M = (ways_to_choose_partition * val_for_M) % MOD
        results.append(final_ans_for_this_M)
        
    sys.stdout.write(" ".join(map(str, results)) + "
")

solve()