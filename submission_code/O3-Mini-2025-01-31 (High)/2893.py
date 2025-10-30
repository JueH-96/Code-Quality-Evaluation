from typing import List
import math

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Initialize best scores based on the parity of the first element (index 0)
        if nums[0] % 2 == 0:
            best_even, best_odd = nums[0], -math.inf
        else:
            best_even, best_odd = -math.inf, nums[0]
        
        # Process remaining elements in order
        for num in nums[1:]:
            if num % 2 == 0:
                # If current number is even, two options:
                # 1. Come from an even number: no penalty.
                # 2. Come from an odd number: penalty of x.
                candidate = max(best_even + num, best_odd + num - x)
                # Update best_even with candidate if better
                best_even = max(best_even, candidate)
            else:
                # If current number is odd:
                # 1. Come from an odd number: no penalty.
                # 2. Come from an even number: penalty of x.
                candidate = max(best_odd + num, best_even + num - x)
                best_odd = max(best_odd, candidate)
                
        # The answer is the best overall score we can obtain (ending in either parity)
        return max(best_even, best_odd)