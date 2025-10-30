import sys
import math
from collections import defaultdict

# Modulo constant
MOD = 998244353

# Function for modular inverse (a^(MOD-2) mod MOD)
def mod_inverse(n):
    return pow(n, MOD - 2, MOD)

# Precompute primes and prime factorizations using a sieve
MAX_VAL_A = 1000
primes = []
# min_prime_factor[i] stores the smallest prime factor of i
# Used for efficient prime factorization
min_prime_factor = [0] * (MAX_VAL_A + 1) 

def sieve():
    for i in range(2, MAX_VAL_A + 1):
        if min_prime_factor[i] == 0: # i is prime
            min_prime_factor[i] = i
            primes.append(i)
        # For each prime p found, mark multiples of p
        for p in primes:
            # Optimization: stop if p is greater than the smallest prime factor of i
            # or if i*p exceeds MAX_VAL_A, to avoid redundant markings
            if p > min_prime_factor[i] or i * p > MAX_VAL_A:
                break
            min_prime_factor[i * p] = p

sieve()

# Function to get prime factorization of a number
def get_prime_factorization(n):
    factors = defaultdict(int)
    if n == 1:
        return factors
    temp_n = n
    while temp_n > 1:
        p = min_prime_factor[temp_n]
        count = 0
        while temp_n % p == 0:
            temp_n //= p
            count += 1
        factors[p] = count
    return factors

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split())) # A_1, ..., A_{N-1}

    # Precompute prime factorizations for all A_i values
    A_factors_list = [get_prime_factorization(a_val) for a_val in A]
    
    # Collect all distinct prime factors involved in A_i values
    all_distinct_prime_factors = set()
    for factors_map in A_factors_list:
        for p in factors_map:
            all_distinct_prime_factors.add(p)

    total_sum_of_scores = 1

    # Main loop: Iterate over each distinct prime factor found in A_i
    for p in all_distinct_prime_factors:
        # dp[current_V_exponent][min_V_exponent] = sum of p^(sum_of_V_exponents_so_far)
        # current_V_exponent (curr_V): Represents V_{i+1}^{(p)}
        # min_V_exponent (min_V): Represents min_{j=1..i+1} V_j^{(p)}
        # The stored value (ways_sum_val) is sum_{paths} p^(sum_{j=1..i+1} V_j^{(p)})
        dp = defaultdict(lambda: defaultdict(int))

        # Base case: Before processing A_1 (i=0), we consider V_1.
        # V_1^{(p)} is defined as 0.
        # So, current_V_exponent = 0.
        # sum_{j=1..1} V_j^{(p)} = V_1^{(p)} = 0.
        # min_{j=1..1} V_j^{(p)} = V_1^{(p)} = 0.
        # The sum of p^(sum_of_V_exponents_so_far) is p^0 = 1.
        dp[0][0] = 1 

        # Iterate through A_i (from A_1 to A_{N-1}), where index i runs from 0 to N-2
        for i in range(N - 1): 
            # Get the exponent of prime p in A_{i+1}
            a_val_p_exp = A_factors_list[i].get(p, 0) 
            
            # new_dp table to store results for V_{i+2}^{(p)}
            new_dp = defaultdict(lambda: defaultdict(int))

            # Iterate over existing states from the current dp table
            for curr_V, min_V_dict in dp.items():
                for min_V, ways_sum_val in min_V_dict.items():
                    if ways_sum_val == 0: # Skip paths with zero contribution
                        continue

                    # If v_p(A_{i+1}) is 0, then b_{i+1}^{(p)} must be 0.
                    # This means V_{i+2}^{(p)} = V_{i+1}^{(p)} + 0 = curr_V.
                    # The minimum V_j^{(p)} up to V_{i+2}^{(p)} will be min(min_V, curr_V).
                    # Since min_V is already the minimum up to V_{i+1}^{(p)}, and V_{i+2}^{(p)} = curr_V,
                    # and by definition min_V <= curr_V, then min(min_V, curr_V) simplifies to min_V.
                    if a_val_p_exp == 0:
                        new_V = curr_V
                        new_min_V = min_V # min_V remains the same if the next term is current V
                        
                        # Accumulate ways_sum_val * p^(new_V) into the new state
                        # ways_sum_val stores sum_{paths} p^(sum_of_V_up_to_V_{i+1}^{(p)})
                        # We are adding p^(V_{i+2}^{(p)}) for each path
                        new_dp[new_V][new_min_V] = (new_dp[new_V][new_min_V] + ways_sum_val * pow(p, new_V, MOD)) % MOD
                    else:
                        # If v_p(A_{i+1}) > 0, there are two choices for b_{i+1}^{(p)}: -a_val_p_exp or a_val_p_exp
                        
                        # Choice 1: b_{i+1}^{(p)} = -a_val_p_exp
                        new_V1 = curr_V - a_val_p_exp
                        new_min_V1 = min(min_V, new_V1) # Update minimum V_j^{(p)} encountered so far
                        new_dp[new_V1][new_min_V1] = (new_dp[new_V1][new_min_V1] + ways_sum_val * pow(p, new_V1, MOD)) % MOD

                        # Choice 2: b_{i+1}^{(p)} = a_val_p_exp
                        new_V2 = curr_V + a_val_p_exp
                        new_min_V2 = min(min_V, new_V2) # Update minimum V_j^{(p)} encountered so far
                        new_dp[new_V2][new_min_V2] = (new_dp[new_V2][new_min_V2] + ways_sum_val * pow(p, new_V2, MOD)) % MOD
            dp = new_dp # Update dp table for the next iteration

        # After processing all A_i (from A_1 to A_{N-1}), the dp table `dp` now represents states for V_N^{(p)}
        sum_for_p = 0
        for curr_V, min_V_dict in dp.items():
            for min_V, ways_sum_val in min_V_dict.items():
                if ways_sum_val == 0: # Skip empty paths
                    continue
                
                # The total exponent for prime p in the score (product of S_j) is:
                # E_p = N * (-min_{j=1..N} V_j^{(p)}) + sum_{j=1..N} V_j^{(p)}
                # `ways_sum_val` already stores the sum over paths of p^(sum_{j=1..N} V_j^{(p)})
                # So we need to multiply by p^(N * (-min_V)) for each path's contribution
                exponent_N_minV = N * (-min_V)
                
                # pow(base, exponent, MOD) correctly handles positive exponents.
                # For negative exponents `E`, it calculates `(base^(-E))^(MOD-2) mod MOD`
                # using modular inverse for negative powers.
                if exponent_N_minV < 0:
                    term_N_minV = pow(mod_inverse(p), -exponent_N_minV, MOD)
                else:
                    term_N_minV = pow(p, exponent_N_minV, MOD)
                
                sum_for_p = (sum_for_p + ways_sum_val * term_N_minV) % MOD
        
        # Multiply this prime's contribution (sum_for_p) into the total sum of scores
        total_sum_of_scores = (total_sum_of_scores * sum_for_p) % MOD

    sys.stdout.write(str(total_sum_of_scores) + "
")

# Call the solve function to run the program
solve()