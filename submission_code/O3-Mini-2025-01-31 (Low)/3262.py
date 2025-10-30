from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array in non-decreasing order.
        nums.sort()
        total = sum(nums)
        
        # Try to use as many segments as possible.
        # We want the condition: (total - longest) > longest
        # If this doesn't hold, remove the largest edge and try again.
        while len(nums) >= 3:
            longest = nums[-1]
            if total - longest > longest:
                return total
            # Remove the largest edge and update total
            total -= longest
            nums.pop()
        return -1