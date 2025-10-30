from typing import List

class Solution:
    MOD = 10**9 + 7
    max_n = 10**5
    fact = [1] * (max_n + 1)
    for i in range(1, max_n + 1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [1] * (max_n + 1)
    inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
    for i in range(max_n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        product = 1
        sum_m = 0
        segments = []
        
        # Left end segment
        if sick[0] > 0:
            m = sick[0]
            segments.append(m)
            sum_m += m
        
        # Middle segments
        for i in range(1, len(sick)):
            prev = sick[i-1]
            curr = sick[i]
            if curr - prev > 1:
                m = curr - prev - 1
                segments.append(m)
                sum_m += m
                product = product * pow(2, m-1, self.MOD) % self.MOD
        
        # Right end segment
        if sick[-1] < n - 1:
            m = (n - 1) - sick[-1]
            segments.append(m)
            sum_m += m
        
        # Compute multinomial coefficient
        if sum_m == 0:
            return 1  # This case is when all children are initially sick, but per problem constraints, this is not possible
        
        multinom = self.fact[sum_m]
        for m in segments:
            multinom = multinom * self.inv_fact[m] % self.MOD
        
        result = (product * multinom) % self.MOD
        return result