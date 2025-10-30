from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        """
        For each prime number p in nums we need the smallest x such that
        x | (x + 1) == p.
        
        Observation:
        Let k be the index of the least–significant 0-bit in x.  
        Then bits 0 … k-1 of x are all 1, bit k is 0.
        Consequently  
            x | (x + 1) = x + 2^k
        and the lower k bits of the result are all 1.
        
        Hence, for a number p to be representable there must exist an index
        k for which
              • bit k of p is 1
              • all lower bits are 1
        (i.e. p ends with k consecutive 1-bits).
        
        Among all such k we must choose the largest one, because
            x = p − 2^k,
        so larger 2^k produces the smaller x.
        
        Algorithm for a single odd prime p:
            1. Count t = number of consecutive 1-bits starting from LSB.
               (t ≥ 1 because p is odd).
            2. Set k = t − 1  (largest index fulfilling the condition).
            3. Answer is x = p − (1 << k).
        
        For p = 2 (the only even prime) no representation exists.
        """
        ans = []
        for p in nums:
            if p == 2:             # even prime → impossible
                ans.append(-1)
                continue

            # count trailing 1-bits
            t = 0
            temp = p
            while temp & 1:
                t += 1
                temp >>= 1

            k = t - 1                      # largest valid index
            x = p - (1 << k)               # minimal x
            ans.append(x)

        return ans