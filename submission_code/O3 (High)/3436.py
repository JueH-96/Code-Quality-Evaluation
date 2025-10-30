from typing import List

class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        # best answer found so far
        best = float('inf')
        
        # set of all OR-values of sub-arrays that end at previous position
        prev: set[int] = set()
        
        for num in nums:
            # all OR values for sub-arrays that end at current index
            curr = {num}
            for v in prev:
                curr.add(v | num)
            
            # update the answer
            for v in curr:
                diff = abs(k - v)
                if diff < best:
                    best = diff
                    if best == 0:                          # cannot do better
                        return 0
            prev = curr                                  
        
        return best