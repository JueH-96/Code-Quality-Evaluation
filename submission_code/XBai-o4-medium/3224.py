from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        # Precompute factorial modulo MOD
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = (fact[i-1] * i) % MOD
        
        intervals = []
        first = sick[0]
        if first > 0:
            intervals.append(('prefix', first))
        
        # Process middle intervals between sick positions
        for i in range(len(sick) - 1):
            current = sick[i]
            next_s = sick[i+1]
            m = next_s - current - 1
            if m > 0:
                intervals.append(('middle', m))
        
        last = sick[-1]
        if last < n - 1:
            m = (n - 1) - last
            intervals.append(('suffix', m))
        
        product_ways = 1
        product_fact = 1
        total_steps = 0
        
        for typ, m in intervals:
            total_steps += m
            product_fact = (product_fact * fact[m]) % MOD
            if typ == 'middle':
                power = m - 1
                product_ways = (product_ways * pow(2, power, MOD)) % MOD
        
        # Compute the answer
        ans = fact[total_steps]
        ans = (ans * product_ways) % MOD
        inv_product_fact = pow(product_fact, MOD - 2, MOD)
        ans = (ans * inv_product_fact) % MOD
        
        return ans