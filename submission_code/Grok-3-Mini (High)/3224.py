import math
from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        MOD = 1000000007
        m = len(sick)
        N_uninf = n - m
        
        # Compute factorial and inverse factorial up to N_uninf
        fact = [1] * (N_uninf + 1)
        for i in range(1, N_uninf + 1):
            fact[i] = (fact[i - 1] * i) % MOD
        
        inv_fact = [1] * (N_uninf + 1)
        for i in range(N_uninf + 1):
            inv_fact[i] = pow(fact[i], MOD - 2, MOD)
        
        # Find gaps and compute E
        gaps = []
        E = 0
        
        # Left gap (end gap)
        len_left = sick[0]
        if len_left > 0:
            gaps.append(len_left)
        
        # Between gaps
        for i in range(m - 1):
            between_len = sick[i + 1] - sick[i] - 1
            if between_len > 0:
                gaps.append(between_len)
                E += (between_len - 1)
        
        # Right gap (end gap)
        right_len = n - sick[-1] - 1
        if right_len > 0:
            gaps.append(right_len)
        
        # Compute product of inverse factorials for gap sizes
        prod_inv_fact = 1
        for size in gaps:
            prod_inv_fact = (prod_inv_fact * inv_fact[size]) % MOD
        
        # Compute multinomial part
        multinomial_part = (fact[N_uninf] * prod_inv_fact) % MOD
        
        # Compute 2^E mod MOD
        two_pow_E = pow(2, E, MOD)
        
        # Compute final answer
        answer = (multinomial_part * two_pow_E) % MOD
        return answer