from typing import List

class Solution:
    def numberOfSequence(self, n: int, sick: List[int]) -> int:
        mod = 10**9 + 7
        
        # Precompute factorials and inverse factorials up to n
        fact = [1] * (n + 1)
        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % mod
        
        invfact = [1] * (n + 1)
        invfact[n] = pow(fact[n], mod - 2, mod)
        for i in range(n, 0, -1):
            invfact[i - 1] = invfact[i] * i % mod
        
        # Build segment lengths
        all_lens = []
        mid_lens = []
        
        # Left end segment (from 0 up to sick[0]-1)
        left_len = sick[0]
        all_lens.append(left_len)
        
        # Middle segments between consecutive sick positions
        for i in range(len(sick) - 1):
            seg_len = sick[i + 1] - sick[i] - 1
            all_lens.append(seg_len)
            # Only nonzero middle segments contribute a 2^(seg_len-1) factor
            if seg_len > 0:
                mid_lens.append(seg_len)
        
        # Right end segment (from sick[-1]+1 up to n-1)
        right_len = (n - 1) - sick[-1]
        all_lens.append(right_len)
        
        # Total number of healthy children to infect
        m = n - len(sick)
        
        # Multinomial factor for interleaving infections across segments:
        # m! / (prod li!)
        ans = fact[m]
        for length in all_lens:
            ans = ans * invfact[length] % mod
        
        # Within each middle segment of length L there are 2^(L-1) ways
        exp = sum(L - 1 for L in mid_lens)
        ans = ans * pow(2, exp, mod) % mod
        
        return ans