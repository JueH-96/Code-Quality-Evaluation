from typing import List

MOD = 10**9 + 7

class Solution:
    def __init__(self):
        max_n = 10**5
        self.fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            self.fact[i] = self.fact[i-1] * i % MOD
        
        self.inv_fact = [1] * (max_n + 1)
        self.inv_fact[max_n] = pow(self.fact[max_n], MOD - 2, MOD)
        for i in range(max_n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % MOD

    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        intervals = []
        first = sick[0]
        if first > 0:
            intervals.append(('prefix', first))
        
        for i in range(1, len(sick)):
            prev = sick[i-1]
            curr = sick[i]
            m = curr - prev - 1
            if m > 0:
                intervals.append(('between', m))
        
        last = sick[-1]
        suffix_len = (n - 1) - last
        if suffix_len > 0:
            intervals.append(('suffix', suffix_len))
        
        prod_2 = 1
        for interval in intervals:
            if interval[0] == 'between':
                m = interval[1]
                prod_2 = (prod_2 * pow(2, m-1, MOD)) % MOD
        
        S = n - len(sick)
        if S == 0:
            return 1 % MOD
        
        multinom = self.fact[S]
        for interval in intervals:
            k_i = interval[1]
            multinom = (multinom * self.inv_fact[k_i]) % MOD
        
        return (prod_2 * multinom) % MOD