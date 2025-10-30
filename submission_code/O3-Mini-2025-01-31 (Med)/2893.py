from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Use very small number to represent negative infinity.
        neg_inf = -10**18  # Should be low enough given constraints.
        
        # Initialize best scores for even and odd last visited element.
        if nums[0] % 2 == 0:
            best_even = nums[0]
            best_odd = neg_inf
        else:
            best_odd = nums[0]
            best_even = neg_inf
        
        # Iterate over the array, starting from index 1
        for num in nums[1:]:
            if num % 2 == 0:
                # If current number is even, we have two ways:
                # 1. Coming from an even last element (no penalty)
                # 2. Coming from an odd last element (with penalty)
                cand_from_even = best_even + num if best_even != neg_inf else neg_inf
                cand_from_odd = best_odd + num - x if best_odd != neg_inf else neg_inf
                current = max(cand_from_even, cand_from_odd)
                # Update best_even with the possibility of not taking current as well.
                best_even = max(best_even, current)
            else:
                # If current number is odd:
                cand_from_odd = best_odd + num if best_odd != neg_inf else neg_inf
                cand_from_even = best_even + num - x if best_even != neg_inf else neg_inf
                current = max(cand_from_odd, cand_from_even)
                best_odd = max(best_odd, current)
        
        return max(best_even, best_odd)