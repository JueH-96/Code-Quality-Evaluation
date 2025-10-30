from typing import List
MOD = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        # Precompute factorial and inverse factorial
        max_n = 2 * n + 2
        factorial = [1] * (max_n)
        for i in range(1, max_n):
            factorial[i] = factorial[i-1] * i % MOD
        inv_fact = [1] * (max_n)
        inv_fact[max_n-1] = pow(factorial[max_n-1], MOD-2, MOD)
        for i in range(max_n-2, -1, -1):
            inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
        # Function to compute binomial coefficient
        def binom(a, b):
            if a < b or b < 0:
                return 0
            return factorial[a] * inv_fact[b] % MOD * inv_fact[a-b] % MOD
        # Precompute Catalan numbers
        def catalan(k):
            if k ==0:
                return 1
            return binom(2*k, k) * pow(k+1, MOD-2, MOD) % MOD
        # Enumerate segments
        segments = []
        sick = sorted(sick)
        prev = -1
        m = len(sick)
        # Left unbounded
        if sick[0] > 0:
            k = sick[0]
            segments.append( ('unbounded', k) )
        # Between sick positions
        for i in range(1, m):
            gap = sick[i] - sick[i-1] -1
            if gap >0:
                segments.append( ('bounded', gap) )
        # Right unbounded
        if sick[-1] < n-1:
            k = n - sick[-1] -1
            segments.append( ('unbounded', k) )
        total_k = 0
        prod_c = 1
        for typ, k in segments:
            total_k += k
            if typ == 'unbounded':
                # Only one way to infect linearly
                pass
            else:
                # Bounded: 2 * Catalan(k-1)
                c = 2 * catalan(k-1) % MOD
                prod_c = prod_c * c % MOD
        # Compute total_fact and product of inv_fact(k_i)
        total_fact = factorial[total_k]
        prod_inv_fact = 1
        for typ, k in segments:
            prod_inv_fact = prod_inv_fact * inv_fact[k] % MOD
        result = prod_c * factorial[total_k] % MOD
        result = result * pow(prod_fact := 1, 1, MOD) % MOD  # dummy step
        result = result * prod_inv_fact % MOD
        return result