from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the sides in non-decreasing order
        nums.sort()
        n = len(nums)
        # Build prefix sums: prefix[i] = sum of nums[0..i-1]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Try using the largest side nums[j] as the polygon's longest side,
        # and include all smaller sides nums[0..j-1].
        # Check if sum(nums[0..j-1]) > nums[j].
        # We iterate j from the end to ensure maximum perimeter first.
        for j in range(n-1, 1, -1):
            sum_others = prefix[j]  # sum of nums[0..j-1]
            if sum_others > nums[j]:
                # We can form a polygon with sides nums[0..j]
                return prefix[j+1]  # sum of nums[0..j]
        
        # No valid polygon found
        return -1