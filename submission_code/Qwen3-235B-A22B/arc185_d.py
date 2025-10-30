MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    inv2 = pow(2, MOD-2, MOD)
    inv6 = pow(6, MOD-2, MOD)
    
    n_mod = N % MOD
    m_mod = M % MOD
    
    # Compute term1
    term1_part_a = (m_mod * ((n_mod - 1 + MOD) % MOD)) % MOD
    term1_part_b = (term1_part_a + 1) % MOD
    term1 = (n_mod * m_mod) % MOD
    term1 = (term1 * term1_part_b) % MOD
    
    # Compute term2_contribution
    A = (n_mod * m_mod + 1) % MOD
    B = ((m_mod - 1 + MOD) % MOD) * m_mod % MOD
    sum_s = B * inv2 % MOD
    term_a = A * sum_s % MOD
    
    C = (2 * m_mod - 1 + MOD) % MOD
    sum_s2_part = B * C % MOD
    sum_s2 = sum_s2_part * inv6 % MOD
    term_b = sum_s2
    
    term2_total = (term_a - term_b + MOD) % MOD
    term2_contribution = (n_mod * term2_total) % MOD
    
    S = (term1 + term2_contribution) % MOD
    print(S)

if __name__ == "__main__":
    main()