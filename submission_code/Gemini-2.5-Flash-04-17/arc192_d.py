import sys
from collections import defaultdict
import math

# Function to compute prime factorization
def prime_factorization(n):
    factors = defaultdict(int)
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors[d] += 1
            temp //= d
        d += 1
    if temp > 1:
        factors[temp] += 1
    return factors

# Modular exponentiation
def power(a, b, m):
    res = 1
    a %= m
    # Handle negative exponents in a specific way for modular inverse
    if b < 0:
        b = b % (m - 1) # By Fermat's Little Theorem
        if b < 0:
            b += m - 1
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % m
        a = (a * a) % m
        b //= 2
    return res

MOD = 998244353

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # 1. Find all unique prime factors of A_i
    prime_factors_set = set()
    A_factors = []
    for a in A:
        factors = prime_factorization(a)
        A_factors.append(factors)
        prime_factors_set.update(factors.keys())

    total_sum_scores = 1

    # 2. DP for each prime factor p
    for p in prime_factors_set:
        # Get exponents alpha_j = v_p(A_j) for A_1, ..., A_{N-1}
        alphas = [factors.get(p, 0) for factors in A_factors]

        # Max possible sum of alphas for this prime p
        # This is used to determine the range of V_i and min_V_i
        # V_i = sum_{j=1}^{i-1} y_j, y_j in {-\alpha_j, \alpha_j}
        # Max V_i is sum_{j=1}^{i-1} \alpha_j
        # Min V_i is -sum_{j=1}^{i-1} \alpha_j
        # Min min_V_i is -sum_{j=1}^{N-1} \alpha_j
        S_max = sum(alphas)

        # DP state: dp[v_shifted][min_v_shifted] = sum of p^{\sum_{k=1}^{i-1} (N-k)y_k}
        # where V_i = v, min_{k=1..i} V_k = min_v
        # Shift = S_max. v_shifted = v + S_max, min_v_shifted = min_v + S_max
        # v range [-S_max, S_max] -> [0, 2*S_max]
        # min_v range [-S_max, 0] -> [0, S_max]

        # dp[v_shifted][min_v_shifted]
        dp = defaultdict(lambda: defaultdict(int))

        # Base case: i=1. V_1 = 0. min_V_1 = 0. Sum of empty product is 1.
        # v_shifted = 0 + S_max, min_v_shifted = 0 + S_max
        dp[0 + S_max][0 + S_max] = 1

        # DP transition for i = 1 to N-1 (choosing y_i)
        # dp[i] is computed from dp[i-1] by choosing y_{i-1}
        # We iterate i from 1 to N-1 to compute dp[i+1] from dp[i] using alpha_i
        # Indices for A are 1..N-1. In code 0..N-2.
        # y_i corresponds to A_i.
        # V_{i+1} = V_i + y_i.
        # dp state index i refers to V_i
        # We have dp for V_i. Use alpha_i (alphas[i-1]) to get to V_{i+1}.
        # This is from V_1 to V_N. N steps.
        # Start with V_1. Compute V_2 using y_1. ... Compute V_N using y_{N-1}.
        # DP table index indicates which V_i value it stores sums for.
        # dp[i] stores sums for V_i. Loop i from 1 to N-1.
        # compute dp[i+1] using dp[i] and alpha_i (alphas[i-1])
        # There are N-1 steps of choosing y_j (j=1 to N-1).
        # V_1 is base case. V_2 needs y_1 ... V_N needs y_{N-1}.
        # Loop N-1 times for y_1 to y_{N-1}.
        # Let's use 0-indexed y_0 .. y_{N-2} for A_0 .. A_{N-2}.
        # V_i = sum_{j=0}^{i-2} y_j. V_1=0. V_2=y_0. V_N = sum y_0..y_{N-2}.
        # DP state index i from 0 to N-1. dp[i] stores sum after choosing y_i.
        # dp[-1] stores base case V_1.
        # dp[i] stores sum after choosing y_i, i=0..N-2. Corresponds to V_{i+2}.
        # DP loop i from 0 to N-2. Choose y_i.
        # dp[i] is sum up to y_{i-1}. Represents V_i.
        # Start: dp state for V_1.
        # Loop i from 0 to N-2 (for A_0 to A_{N-2})
        # In iteration i, we choose y_{i+1}. Index into A is i. alpha = alphas[i].
        # Exponent (N-(i+1)).

        # dp_current stores values for V_i. Initially V_1.
        dp_current = defaultdict(lambda: defaultdict(int))
        dp_current[0 + S_max][0 + S_max] = 1

        # Loop N-1 times, choosing y_1 ... y_{N-1}
        for i in range(N - 1): # Corresponds to choosing y_{i+1}, using alpha_{i+1} = alphas[i]
            dp_next = defaultdict(lambda: defaultdict(int))
            alpha = alphas[i] # alpha_{i+1} for A_{i+1}

            if alpha == 0:
                y = 0
                p_y_exp = power(p, (N - 1 - i) * y, MOD) 
                
                for v_shifted, min_v_dict in dp_current.items():
                    v = v_shifted - S_max
                    v_next = v + y
                    
                    for min_v_shifted, score_sum in min_v_dict.items():
                         min_v = min_v_shifted - S_max
                         min_v_next = min(min_v, v_next)
                         dp_next[v_next + S_max][min_v_next + S_max] = (dp_next[v_next + S_max][min_v_next + S_max] + score_sum * p_y_exp) % MOD

            else:
                # choices y = alpha or y = -alpha
                y_alpha = alpha
                y_neg_alpha = -alpha
                p_y_alpha_exp = power(p, (N - 1 - i) * y_alpha, MOD)
                p_y_neg_alpha_exp = power(p, (N - 1 - i) * y_neg_alpha, MOD)

                for v_shifted, min_v_dict in dp_current.items():
                    v = v_shifted - S_max
                    
                    # Choice y = alpha
                    v_next_alpha = v + y_alpha
                    
                    for min_v_shifted, score_sum in min_v_dict.items():
                         min_v = min_v_shifted - S_max
                         min_v_next_alpha = min(min_v, v_next_alpha)
                         dp_next[v_next_alpha + S_max][min_v_next_alpha + S_max] = (dp_next[v_next_alpha + S_max][min_v_next_alpha + S_max] + score_sum * p_y_alpha_exp) % MOD
                    
                    # Choice y = -alpha
                    v_next_neg_alpha = v + y_neg_alpha
                    
                    for min_v_shifted, score_sum in min_v_dict.items():
                         min_v = min_v_shifted - S_max
                         min_v_next_neg_alpha = min(min_v, v_next_neg_alpha)
                         dp_next[v_next_neg_alpha + S_max][min_v_next_neg_alpha + S_max] = (dp_next[v_next_neg_alpha + S_max][min_v_next_neg_alpha + S_max] + score_sum * p_y_neg_alpha_exp) % MOD

            dp_current = dp_next

        # After N-1 steps (y_1 to y_{N-1}), dp_current stores values for V_N.
        # Calculate S_p from dp_current
        S_p = 0
        for v_shifted, min_v_dict in dp_current.items():
            v = v_shifted - S_max
            for min_v_shifted, score_sum in min_v_dict.items():
                min_v = min_v_shifted - S_max
                
                # Score exponent for this choice sequence: N * max(0, -min_v) + sum_{j=1}^{N-1}(N-j)y_j
                # score_sum in dp_current is p^{sum_{j=1}^{N-1}(N-j)y_j}
                S_p_term = (score_sum * power(p, N * max(0, -min_v), MOD)) % MOD
                S_p = (S_p + S_p_term) % MOD

        total_sum_scores = (total_sum_scores * S_p) % MOD

    print(total_sum_scores)

solve()