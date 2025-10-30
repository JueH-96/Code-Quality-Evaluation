from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # We will keep track of the maximum score we can achieve if the last picked value is even or odd.
        # dp_even: max score ending with an even number
        # dp_odd: max score ending with an odd number
        
        # Initialize dp_even and dp_odd with very small numbers
        dp_even = float('-inf')
        dp_odd = float('-inf')
        
        # Base case: we start at position 0
        if nums[0] % 2 == 0:
            dp_even = nums[0]
        else:
            dp_odd = nums[0]
        
        # Iterate over the remaining elements
        for i in range(1, len(nums)):
            if nums[i] % 2 == 0:
                # If nums[i] is even:
                # We can come from an even dp (no penalty) or from an odd dp (penalty x)
                new_even = max(dp_even + nums[i], dp_odd + nums[i] - x)
                dp_even = new_even
            else:
                # If nums[i] is odd:
                # We can come from an odd dp (no penalty) or from an even dp (penalty x)
                new_odd = max(dp_odd + nums[i], dp_even + nums[i] - x)
                dp_odd = new_odd
        
        # The answer is the maximum between dp_even and dp_odd
        return max(dp_even, dp_odd)