import sys

sys.setrecursionlimit(10**4) # Adjusted recursion limit (higher if needed)

MOD = 998244353

# Precompute primes up to sqrt(N_max) = 10^5
MAX_SQRT_N = 10**5 # Max value for sqrt(N)
primes = []
is_prime_sieve = [True] * (MAX_SQRT_N + 1)

def sieve():
    is_prime_sieve[0] = is_prime_sieve[1] = False
    for i in range(2, int(MAX_SQRT_N**0.5) + 1):
        if is_prime_sieve[i]:
            for multiple in range(i*i, MAX_SQRT_N + 1, i):
                is_prime_sieve[multiple] = False
    for i in range(2, MAX_SQRT_N + 1):
        if is_prime_sieve[i]:
            primes.append(i)

memo_dfs = {}

# MAX_E: max exponent. log2(10^10) ~ 33.2. Max e around 34 for p=2.
# For p=3, 3^20 ~ 3.5e9, 3^21 > 10^10. Max e around 20.
# Set MAX_E to a safe general upper bound like 60.
MAX_E = 60 
fact = [1] * (MAX_E + 1)

def precompute_factorials():
    for i in range(1, MAX_E + 1):
        fact[i] = (fact[i-1] * i) % MOD

def nCr_mod(n_val, r_val):
    if r_val < 0 or r_val > n_val: # r_val is exponent e, n_val is M-1+e. M>=1, e>=0. So n_val >= r_val is always true.
        return 0
    if r_val == 0: 
        return 1
    
    num = 1
    for i in range(r_val):
        term = (n_val - i) % MOD
        num = num * term % MOD
    
    den = fact[r_val]
    
    return num * pow(den, MOD - 2, MOD) % MOD

def get_binom_val(e_val, M_val):
    return nCr_mod(M_val - 1 + e_val, e_val)

# calc_type: 0 for H_M(N) (total sum of d_M(k)), 1 for G(N) (sum of d_M(k) for non-good k)
def dfs(prime_idx, limit_n, M_val, calc_type):
    state = (prime_idx, limit_n, calc_type)
    if state in memo_dfs:
        return memo_dfs[state]
    
    res = 1 # Base case: contribution for k=1 (product of empty set of primes from primes[prime_idx:])
            # d_M(1) = 1. sigma(1)=1, so 1 is non-good.
            # So f_0(1)=1, f_1(1)=1. This '1' is correct for both calc_types.
    
    if prime_idx >= len(primes): # No more primes from our list (up to MAX_SQRT_N)
        memo_dfs[state] = res
        return res

    # Loop through primes starting from primes[prime_idx]
    for i in range(prime_idx, len(primes)):
        p = primes[i]

        if p > limit_n: # Current prime p is already greater than remaining limit.
            break
        
        # Contribution from numbers whose smallest prime factor (from current list of primes) is p
        # These are p^e * K, where K's smallest prime factor is > p (or K=1)
        
        pe = p # current p^e value, starting with p^1
        for e in range(1, MAX_E + 1): # Max exponent e for p
            if pe > limit_n: # p^e exceeds limit
                break

            coeff = get_binom_val(e, M_val)
            
            if calc_type == 1: # G(N) sum, check if p^e is a "non-good part"
                # A part p^e is "non-good part" if sigma(p^e) % 3 != 0
                is_prime_power_non_good_part = True
                if p == 3: # sigma(3^e) % 3 == 1, always a non-good part.
                    pass
                elif p % 3 == 1: # p = 1 (mod 3)
                    if e % 3 == 2: # e = 2 (mod 3) makes sigma(p^e)%3 == 0 ("good part")
                        is_prime_power_non_good_part = False
                else: # p = 2 (mod 3) (includes p=2 itself)
                    if e % 2 == 1: # e is odd makes sigma(p^e)%3 == 0 ("good part")
                        is_prime_power_non_good_part = False
                
                if not is_prime_power_non_good_part:
                    coeff = 0 
            
            if coeff != 0:
                 term_val_recursive = dfs(i + 1, limit_n // pe, M_val, calc_type)
                 res = (res + coeff * term_val_recursive) % MOD
            
            if pe > limit_n // p : # Avoid overflow for pe * p, and check if next power fits
                break
            pe *= p
            
    memo_dfs[state] = res
    return res

def main_solve():
    N_in, M_in = map(int, sys.stdin.readline().split())

    sieve()
    precompute_factorials()
    
    memo_dfs.clear() # Clear memo for first DFS call
    total_sum_dmk = dfs(0, N_in, M_in, 0)
    
    memo_dfs.clear() # Clear memo for second DFS call
    non_good_sum_dmk = dfs(0, N_in, M_in, 1)
    
    ans = (total_sum_dmk - non_good_sum_dmk + MOD) % MOD
    print(ans)

if __name__ == '__main__':
    main_solve()