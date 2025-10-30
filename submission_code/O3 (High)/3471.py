from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        # Sort the numbers so that the smallest elements are at the beginning
        # and the largest at the end.  We will pair them off from both ends.
        nums.sort()
        
        left, right = 0, len(nums) - 1
        min_avg = float('inf')
        
        # Each iteration pairs one smallest with one largest (n // 2 iterations)
        while left < right:
            avg = (nums[left] + nums[right]) / 2.0
            if avg < min_avg:
                min_avg = avg
            left += 1
            right -= 1
        
        return min_avg