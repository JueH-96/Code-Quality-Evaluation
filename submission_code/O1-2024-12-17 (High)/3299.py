from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        
        # We'll keep track of the best (maximum) answer found
        ans = 0
        
        # Special handling for x = 1, because 1^(2^k) = 1 for all k,
        # so all selected elements would be 1. In such a pattern of length L,
        # we need L copies of 1, with all but one used in pairs (2 per exponent),
        # therefore the length must be odd and ≤ freq[1].
        # The maximum odd integer ≤ freq[1] is 2*((freq[1] - 1)//2) + 1.
        if 1 in freq:
            count_1 = freq[1]
            # Largest odd integer <= count_1
            best_for_1 = 2*((count_1 - 1)//2) + 1
            ans = max(ans, best_for_1)
        
        # Now handle x != 1
        for x in freq:
            if x == 1:
                continue
            
            # If there's at least one x, we can start with a chain of length 1
            # (just [x]), provided freq[x] >= 1
            if freq[x] < 1:
                continue
            
            # Build the list of consecutive powers: x^(1), x^(2), x^(4), x^(8)...
            # We stop when exceeding 1e9 or if that power doesn't appear in freq at all
            # (since skipping exponents is not allowed by the pattern).
            powers = [x]
            while True:
                nxt = powers[-1] * powers[-1]
                if nxt > 10**9 or freq.get(nxt, 0) == 0:
                    break
                powers.append(nxt)
                # Safety limit; beyond 30 or so won't matter because
                # x^(2^30) is huge for even moderately large x.
                if len(powers) > 30:
                    break
            
            # At least freq[x] >= 1, so we can form a chain of length 1
            best_chain_length = 1
            
            # Count how many (from the start) have freq >= 2
            # Those can serve as the "double usage" exponents in the pattern.
            p = 0
            nP = len(powers)
            while p < nP and freq[powers[p]] >= 2:
                p += 1
            
            # If all powers[] have freq >= 2, then the largest exponent is powers[nP-1],
            # used once. Pattern length = 2*(nP-1) + 1 = 2nP - 1.
            if p == nP:
                best_chain_length = 2*nP - 1
            
            else:
                # Otherwise, we can use exponents [0..p-1] as double-usage,
                # and try to use powers[p] as the single-usage (largest exponent) if freq >= 1.
                # Chain length = 2*p + 1 if freq[powers[p]] >= 1,
                # or fallback to 2*(p-1) + 1 if p>0,
                # or 1 if we cannot extend further.
                
                # Always can form length 1 from powers[0] if freq[x]>=1
                candidate1 = 1  
                
                # Try using powers[p] as the largest exponent (need freq >=1)
                candidate2 = 2*p + 1 if (p < nP and freq[powers[p]] >= 1) else 0
                
                # Or use powers[p-1] if p>0 as the largest exponent
                candidate3 = 2*(p - 1) + 1 if p > 0 else 0
                
                best_chain_length = max(candidate1, candidate2, candidate3)
            
            ans = max(ans, best_chain_length)
        
        return ans