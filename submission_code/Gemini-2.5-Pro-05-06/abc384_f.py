import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Max k for A_i = 2^k * m_i. A_i <= 10^7 implies 2^k <= 10^7. 2^23 ~ 8*10^6. Max k is 23.
    MAX_K_A = 23 
    # Max p for m_i+m_j = 2^p * odd. m_i, m_j <= 10^7. m_i+m_j <= 2*10^7. 2^p <= 2*10^7. 2^24 ~ 1.6*10^7. Max p is 24.
    MAX_K_M = 24 

    m_values = [0] * N
    k_values = [0] * N

    for i in range(N):
        val = A[i]
        # (val & -val) is the lowest set bit value (a power of 2, say 2^k_val)
        # k_val = (val & -val).bit_length() - 1
        # m_val = val / (val & -val)
        lowest_set_bit_power = (val & -val) 
        k_val = lowest_set_bit_power.bit_length() - 1
        m_val = val // lowest_set_bit_power
        
        k_values[i] = k_val
        m_values[i] = m_val

    total_sum = 0
    for i in range(N):
        total_sum += m_values[i] # Contribution from f(A_i+A_i) = f(A_i) = m_i

    # Data structures for sum over i < j
    # For k_i != k_j part:
    Sm = [0] * (MAX_K_A + 1) # Sm[k] = sum of m_x for A_x processed so far (index < current j) with v_2(A_x)=k
    Cnt = [0] * (MAX_K_A + 1)# Cnt[k] = count of A_x processed so far with v_2(A_x)=k

    # For k_i = k_j part:
    # stats[k][p_loop] is a dictionary: residue_mod_M -> (count, sum_of_m_values)
    # k is v_2(A_i)
    # p_loop = p-1, where p = v_2(m_i+m_j) ranges from 1 to MAX_K_M
    stats = [[{} for _ in range(MAX_K_M)] for _ in range(MAX_K_A + 1)]

    # Precompute powers of 2
    # Max exponent needed: for k_i != k_j, it's MAX_K_A. For k_i = k_j, modulus is 2^(p+1), so MAX_K_M+1.
    # Overall max exponent is max(MAX_K_A, MAX_K_M+1). Max is MAX_K_M+1 = 25.
    max_exponent_needed = max(MAX_K_A, MAX_K_M + 1)
    powers_of_2 = [1] * (max_exponent_needed + 1) # powers_of_2[i] will be 2^i
    for i in range(1, max_exponent_needed + 1):
        powers_of_2[i] = powers_of_2[i-1] * 2

    for j in range(N):
        mj, kj = m_values[j], k_values[j]

        # Part 1: Iterate $i<j$, $k_i 
eq k_j$. $A_i$ are elements processed before $A_j$.
        for k_other in range(MAX_K_A + 1):
            if Cnt[k_other] == 0: # No elements with this k_other processed yet
                continue
            if k_other == kj: # This case handled by Part 2
                continue
            
            if k_other < kj: # k_i < k_j
                # Sum for these (A_i, A_j) pairs is m_i + 2^(kj-k_other)*mj
                # Sum over all such i: Sm[k_other] + mj * powers_of_2[kj - k_other] * Cnt[k_other]
                total_sum += Sm[k_other] + mj * powers_of_2[kj - k_other] * Cnt[k_other]
            else: # k_other > kj (so k_i > k_j)
                # Sum for these (A_i, A_j) pairs is 2^(k_other-kj)*m_i + mj
                # Sum over all such i: powers_of_2[k_other - kj] * Sm[k_other] + mj * Cnt[k_other]
                total_sum += powers_of_2[k_other - kj] * Sm[k_other] + mj * Cnt[k_other]

        # Part 2: Iterate $i<j$, $k_i = k_j = kj$.
        # Term is f(m_i+m_j) = (m_i+m_j)/2^p, where p = v_2(m_i+m_j).
        # Condition: m_i = (2^p - m_j) (mod 2^(p+1)).
        for p_loop in range(MAX_K_M): # p_loop from 0 to MAX_K_M-1
            p_val = p_loop + 1 # p (v_2(m_i+m_j)) from 1 to MAX_K_M
            
            modulus_M = powers_of_2[p_val + 1] # This is 2^(p+1)
            term_2_p = powers_of_2[p_val]   # This is 2^p

            # Target residue R for m_i
            R = (term_2_p - mj) % modulus_M # Python's % is fine for negative numerators
            
            current_k_p_stats = stats[kj][p_loop] # Dict for fixed k=kj and fixed p
            if R in current_k_p_stats:
                count_R, sum_R = current_k_p_stats[R]
                # Sum for these m_i is (m_i+mj)/2^p. Aggregated: (sum_R + mj*count_R)/2^p
                total_sum += (sum_R + mj * count_R) // term_2_p
        
        # Update data structures with (mj, kj) from current A_j
        Sm[kj] += mj
        Cnt[kj] += 1

        for p_loop in range(MAX_K_M): 
            p_val = p_loop + 1 
            modulus_M = powers_of_2[p_val + 1]
            
            # Residue of mj modulo M=2^(p+1)
            rj = mj % modulus_M

            current_k_p_stats = stats[kj][p_loop]
            current_count, current_sum = current_k_p_stats.get(rj, (0,0))
            current_k_p_stats[rj] = (current_count + 1, current_sum + mj)
            
    sys.stdout.write(str(total_sum) + "
")

solve()