from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the side lengths in non-decreasing order
        nums.sort()
        # total_sum holds the sum of the current candidate sides
        total_sum = sum(nums)
        n = len(nums)
        
        # We try using the longest side at index i as the max side.
        # If sum of the rest > nums[i], we have a valid polygon.
        # Otherwise, drop nums[i] and try with the next smaller as the max side.
        for i in range(n - 1, 1, -1):
            if total_sum - nums[i] > nums[i]:
                # We can form a polygon with sides nums[0..i], perimeter = total_sum
                return total_sum
            # Otherwise, exclude nums[i] from consideration
            total_sum -= nums[i]
        
        # If no valid polygon is found
        return -1