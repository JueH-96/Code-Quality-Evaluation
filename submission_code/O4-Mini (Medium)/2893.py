from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # best_even: best score ending at an even-valued position seen so far
        # best_odd:  best score ending at an odd-valued position seen so far
        best_even = float("-inf")
        best_odd = float("-inf")
        
        # Initialize with position 0
        if nums[0] % 2 == 0:
            best_even = nums[0]
        else:
            best_odd = nums[0]
        
        # Track the overall maximum
        res = nums[0]
        
        # Process positions 1..n-1
        for v in nums[1:]:
            if v % 2 == 0:
                # If we come from an even, no penalty; from odd, penalty x
                dp = v + max(best_even, best_odd - x)
                best_even = max(best_even, dp)
            else:
                # If we come from an odd, no penalty; from even, penalty x
                dp = v + max(best_odd, best_even - x)
                best_odd = max(best_odd, dp)
            
            res = max(res, dp)
        
        return res