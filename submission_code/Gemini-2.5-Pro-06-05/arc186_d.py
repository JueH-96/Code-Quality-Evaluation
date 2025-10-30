import sys

# Increase recursion limit for safety, though the solution is iterative.
sys.setrecursionlimit(10**6)

def solve():
    """
    Main function to solve the Polish sequence counting problem.
    """
    try:
        input = sys.stdin.readline
        N = int(input())
        if N == 0:
            print(0)
            return
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handle potential empty input on some platforms
        return

    MOD = 998244353

    # Precomputation for combinations
    MAX_FACT = 2 * N + 5
    fact = [1] * (MAX_FACT + 1)
    inv_fact = [1] * (MAX_FACT + 1)
    for i in range(1, MAX_FACT + 1):
        fact[i] = (fact[i - 1] * i) % MOD

    inv_fact[MAX_FACT] = pow(fact[MAX_FACT], MOD - 2, MOD)
    for i in range(MAX_FACT - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    def nCr_mod(n, r):
        """Computes nCr modulo MOD."""
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % MOD
        return (num * den) % MOD

    def get_catalan(n):
        """Computes the n-th Catalan number."""
        if n < 0: return 0
        if n == 0: return 1
        return (nCr_mod(2 * n, n) * pow(n + 1, MOD - 2, MOD)) % MOD

    def count_ways(L, C):
        """Calculates the number of ways to form a valid suffix."""
        if L == 0:
            return 1 if C == 1 else 0
        if L - 1 + C < 0 or C > 1:
            return 0
        if C == 1:
            return get_catalan(L - 1)
        
        n, k = 2 * L + C - 2, L - 1
        if k < 0 or k > n: return 0
        
        term1 = nCr_mod(n, k)
        term2 = (1 - C + MOD) % MOD
        num = (term1 * term2) % MOD
        inv_L = pow(L, MOD - 2, MOD)
        return (num * inv_L) % MOD

    def sum_C(n, k, d):
        """Computes sum_{i=0 to d} C(n-i, k)."""
        if d < 0: return 0
        return (nCr_mod(n + 1, k + 1) - nCr_mod(n - d, k + 1) + MOD) % MOD

    # Identity: sum_{i=0..d} (A+i)*C(N-i, K) = (A+N-K)*S1(N,K,d) - (K+1)*S1(N,K+1,d)
    def sum_A_plus_i_C(A, N, K, d):
        if d < 0: return 0
        s1 = sum_C(N, K, d)
        s2 = sum_C(N, K + 1, d)
        term1 = ((A + N - K) % MOD * s1) % MOD
        term2 = ((K + 1) * s2) % MOD
        return (term1 - term2 + MOD) % MOD
        
    total_count = 0
    current_sum = 0
    is_A_ok = True

    for i in range(N):
        p_len = i + 1
        suffix_len = N - p_len
        v_end = A[i] - 1

        if v_end >= 0:
            if p_len < N:
                v_min = p_len - current_sum
                if v_min <= v_end:
                    C_base = p_len - current_sum
                    
                    # Case C = 1 => v = C_base - 1
                    if v_min <= C_base - 1 <= v_end:
                        total_count = (total_count + count_ways(suffix_len, 1)) % MOD

                    # Case C <= 0 => v >= C_base
                    v_start_C_le_0 = max(v_min, C_base)
                    
                    if v_start_C_le_0 <= v_end:
                        L = suffix_len
                        if L > 0:
                            K = L - 1
                            v_start, d = v_start_C_le_0, v_end - v_start_C_le_0
                            A_const = 1 - C_base + v_start
                            N_prime = 2*L + C_base - 2 - v_start
                            
                            # sum (1-C_base+v) * C(2L+C_base-2-v, K)
                            sum_val = sum_A_plus_i_C(A_const, N_prime, K, d)
                            
                            inv_L = pow(L, MOD-2, MOD)
                            total_count = (total_count + sum_val * inv_L) % MOD
            else: # p_len == N
                v_target = N - 1 - current_sum
                if 0 <= v_target <= v_end:
                    total_count = (total_count + 1) % MOD

        current_sum += A[i]
        if i < N - 1:
            if current_sum < p_len:
                is_A_ok = False
                break
        else:
            if current_sum != N - 1:
                is_A_ok = False
    
    if is_A_ok:
        total_count = (total_count + 1) % MOD
        
    print(total_count)

solve()