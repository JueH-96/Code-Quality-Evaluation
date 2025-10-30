import sys
from collections import defaultdict

MOD = 998244353

def power(a, b):
    res = 1
    base = a % MOD
    if b == 0:
        return 1
    if base == 0:
        # Should not happen for prime base p in this problem
        return 0 if b > 0 else 0 # Or error for b < 0
        
    if b < 0:
        b = -b
        base = pow(base, MOD - 2, MOD) # Modular inverse using Fermat's Little Theorem
        
    while b > 0:
        if b % 2 == 1:
            res = (res * base) % MOD
        base = (base * base) % MOD
        b //= 2
    return res

def get_prime_factors(n):
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

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    all_prime_factors = set()
    A_factors = []
    for val in A:
        factors = get_prime_factors(val)
        A_factors.append(factors)
        all_prime_factors.update(factors.keys())

    total_sum_score = 1

    for p in all_prime_factors:
        # a[i] is the exponent of p in A[i] (0-indexed for A list)
        a = [A_factors[i].get(p, 0) for i in range(N - 1)]

        # dp maps (e, m) -> sum of p^W
        # where e = e_{i+1}, m = m_{i+1}, W = sum_{j=0}^{i} (N-1-j)delta_j a_j
        # Index i represents processing A[i] (0-indexed)
        # dp state is after processing A[i]
        
        # Initial state before processing A[0]: e_1=0, m_1=0, W_1=0
        # Let dp represent states for (e_{i+1}, m_{i+1}) after processing A[i]
        # dp[0] is after processing A[-1] (empty set), state is (e_1, m_1)
        
        dp = { (0, 0): 1 } # Map (e, m) -> sum of p^W. Represents state for e_1, m_1, W_1

        for i in range(N - 1): # Process A[i] (index i, exponent a[i])
            next_dp = defaultdict(int)
            cur_a = a[i] # exponent a_{i,p} for A[i]

            # cur_e, cur_m are e_{i+1}, m_{i+1} from previous step
            for (cur_e, cur_m), val in dp.items():
                # delta_i for A[i]
                
                # If a[i] is 0, delta_i must be effectively 0.
                if cur_a == 0:
                    next_e = cur_e
                    next_m = cur_m # min(cur_m, cur_e), since cur_m <= cur_e always true
                    weight = 0
                    # pow_val = power(p, weight) # always 1
                    next_dp[(next_e, next_m)] = (next_dp[(next_e, next_m)] + val) % MOD
                else: # cur_a > 0, delta_i can be 1 or -1
                    # Option 1: delta_i = 1 (exponent a[i] from A[i] goes to Q_{i+1})
                    # e_{i+2} = e_{i+1} + a_i
                    next_e1 = cur_e + cur_a
                    # m_{i+2} = min(m_{i+1}, e_{i+2})
                    next_m1 = min(cur_m, next_e1)
                    # W_{i+2} = W_{i+1} + (N-1-(i)) delta_i a_i
                    # Coefficient for delta_i a_i is (N-1-i)
                    weight1 = (N - 1 - i) * cur_a
                    pow_val1 = power(p, weight1)
                    next_dp[(next_e1, next_m1)] = (next_dp[(next_e1, next_m1)] + val * pow_val1) % MOD

                    # Option 2: delta_i = -1 (exponent a[i] from A[i] goes to P_{i+1})
                    next_e2 = cur_e - cur_a
                    next_m2 = min(cur_m, next_e2)
                    weight2 = (N - 1 - i) * (-cur_a)
                    pow_val2 = power(p, weight2)
                    next_dp[(next_e2, next_m2)] = (next_dp[(next_e2, next_m2)] + val * pow_val2) % MOD
            
            dp = next_dp # dp now stores states for (e_{i+2}, m_{i+2}) after processing A[i]

        # After the loop (i goes from 0 to N-2), dp contains entries (e_N, m_N) -> sum of p^{W_N} for different delta choices
        prime_sum_score = 0
        # dp contains states (e_N, m_N) after processing A[N-2]
        for (final_e, final_m), val in dp.items():
            # Exponent of p in score = N * max(0, -m_N) + W_N
            # W_N is captured by val's power component
            exponent_term = N * max(0, -final_m)
            prime_score_part = power(p, exponent_term)
            prime_sum_score = (prime_sum_score + val * prime_score_part) % MOD

        total_sum_score = (total_sum_score * prime_sum_score) % MOD

    print(total_sum_score)

solve()