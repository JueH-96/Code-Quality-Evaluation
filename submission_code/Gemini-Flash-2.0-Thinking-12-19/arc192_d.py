import sys
from collections import defaultdict

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def prime_factorization(n):
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors[d] = factors.get(d, 0) + 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors

def get_primes(A):
    primes = set()
    for a in A:
        factors = prime_factorization(a)
        primes.update(factors.keys())
    return sorted(list(primes))

MOD = 998244353

def modular_power(base, exp, modulus):
    if exp == 0:
        return 1
    if exp < 0:
        base_inv = modular_power(base, modulus - 2, modulus) # Fermat's Little Theorem
        return modular_power(base_inv, -exp, modulus)
    
    res = 1
    base %= modulus
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % modulus
        base = (base * base) % modulus
        exp //= 2
    return res

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    primes = get_primes(A)
    
    total_score_sum = 1

    for p in primes:
        a_p = [prime_factorization(a).get(p, 0) for a in A]
        
        S_p_max = sum(a_p) # Max possible absolute value of e_i or sum of e_k
        # e_i = sum_{j=1}^{i-1} c_j, c_j in {a_j, -a_j, 0}
        # max |e_i| <= sum_{j=1}^{i-1} a_p[j-1] <= S_p_max
        # The range of e_i is [-S_p_max, S_p_max].
        # Offset for DP states: S_p_max

        # DP state: dp[i][cur_e_offset][min_e_offset][max_neg_e] = sum of p^{\sum_{k=1}^i e_k}
        # i: corresponds to e_i (from 1 to N)
        # cur_e_offset: e_i + S_p_max (0 to 2*S_p_max)
        # min_e_offset: min_{k=1..i} e_k + S_p_max (0 to S_p_max)
        # max_neg_e: max_{k=1..i} (-e_k) (0 to S_p_max)

        dp = [defaultdict(int), defaultdict(int)]
        
        # Base case i=1, corresponds to e_1
        # e_1 = 0. cur_e_offset = S_p_max. min_e = 0. max_neg_e = 0. min_e_offset = S_p_max.
        # Sum of e_k up to e_1 is e_1 = 0.
        dp[1 % 2][(S_p_max, S_p_max, 0)] = 1 # p^0 = 1

        # Iterate i from 1 to N-1. Calculate e_{i+1} from e_i and c_i.
        for i in range(1, N): # Transition for c_i, calculating e_{i+1} from e_i
            curr_dp = dp[i % 2]
            next_dp = dp[(i + 1) % 2]
            next_dp.clear()

            if a_p[i-1] == 0:
                c_values = [0]
            else:
                c_values = [a_p[i-1], -a_p[i-1]]

            for (cur_e_offset, min_e_offset, max_neg_e), val in curr_dp.items():
                e_i = cur_e_offset - S_p_max # Actual e_i value

                for c_i in c_values:
                    e_i_plus_1 = e_i + c_i

                    # Calculate new state variables (with offset)
                    cur_e_plus_1_offset = e_i_plus_1 + S_p_max
                    min_e_plus_1_offset = min(min_e_offset, cur_e_plus_1_offset)
                    max_neg_e_plus_1 = max(max_neg_e, -e_i_plus_1)

                    # The value 'val' is the sum of p^(sum_{k=1..i} e_k) over paths ending here
                    # We need to update this sum to include e_{i+1}
                    # The new value is sum of p^(sum_{k=1..i} e_k + e_{i+1}) = sum of p^(sum_{k=1..i} e_k) * p^e_{i+1}
                    
                    next_dp[(cur_e_plus_1_offset, min_e_plus_1_offset, max_neg_e_plus_1)] = \
                        (next_dp[(cur_e_plus_1_offset, min_e_plus_1_offset, max_neg_e_plus_1)] + val * modular_power(p, e_i_plus_1, MOD)) % MOD

        # After iterating up to i=N-1, we have states for e_N in dp[N%2]
        # The states represent (e_N, min_{k=1..N} e_k, max_{k=1..N} (-e_k))
        # The value is sum of p^(sum_{k=1..N} e_k)

        sum_p_scores = 0
        final_dp = dp[N % 2]

        for (e_N_offset, min_e_offset, max_neg_e), val in final_dp.items():
            # Actual values
            min_e = min_e_offset - S_p_max
            max_neg_e_actual = max_neg_e

            # Check validity condition: max(0, max_neg_e_actual) + min_e = 0
            is_valid = False
            if min_e >= 0:
                 if min_e == 0 and max_neg_e_actual == 0: # If min_e=0, implies all e_k >= 0. max_neg_e = max(-e_k) = -min(e_k) = -0 = 0.
                     is_valid = True
            else: # min_e < 0
                if max_neg_e_actual == -min_e:
                     is_valid = True
            
            if is_valid:
                # Score contribution for this prime p for this specific path
                # p^{N \cdot \max(0, \max_{k=1..N}(-e_k)) + \sum_{k=1..N} e_k}
                # val is sum of p^(sum e_k) over paths ending in this state
                # We need to multiply val by p^(N * max(0, max_neg_e_actual))

                exponent_term = N * max(0, max_neg_e_actual)

                term = (val * modular_power(p, exponent_term, MOD)) % MOD
                sum_p_scores = (sum_p_scores + term) % MOD

        total_score_sum = (total_score_sum * sum_p_scores) % MOD

    print(total_score_sum)

solve()