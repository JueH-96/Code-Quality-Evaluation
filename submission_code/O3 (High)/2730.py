from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        """
        The optimal way to use the operations is to apply all of them
        to a single element.  If an optimal solution used the operations
        on several elements, shifting all of them onto the element that
        currently owns the highest set-bit would move that bit further
        to the left and strictly increase the overall OR, a contradiction.
        
        Hence for every index i we only have to evaluate:
              (nums[i] << k) | OR_of_all_other_numbers
        and take the maximum.
        
        The OR of “all other numbers” can be obtained in O(1) with
        prefix / suffix OR arrays, giving an overall O(n) algorithm.
        """
        n = len(nums)
        
        # prefix[i]  = OR of nums[0 .. i-1]
        # suffix[i]  = OR of nums[i .. n-1]
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] | nums[i]
        
        suffix = [0]*(n+1)
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] | nums[i]
        
        best = 0
        for i in range(n):
            # OR of everything except nums[i]
            others = prefix[i] | suffix[i+1]
            cand   = (nums[i] << k) | others
            best   = max(best, cand)
        
        return best