from typing import List
from collections import Counter

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        cnt = Counter(nums)                       # frequency of every value
        best = 1                                  # we can always pick one element
        
        # special treatment for value 1 (1² = 1, hence squaring does not create a new number)
        ones = cnt.get(1, 0)
        if ones:                                  # the longest valid sequence made only of 1-s
            best = max(best, ones if ones & 1 else ones - 1)   # largest odd ≤ ones
        
        # treat every other value as the possible base x (x > 1)
        for x in cnt:
            if x == 1:
                continue                          # already handled
            length = 1                            # current sequence length (only x itself)
            val = x                               # current central value  (x, x², x⁴, …)
            
            # try to extend the chain step by step
            while cnt[val] >= 2:                  # the former centre must now appear on both sides
                nxt = val * val                   # next central value (square of current)
                if cnt.get(nxt, 0) == 0:          # new centre not available – stop
                    break
                length += 2                       # we add two elements each step
                best = max(best, length)          # update answer
                val = nxt                         # continue with the new centre
        
        return best