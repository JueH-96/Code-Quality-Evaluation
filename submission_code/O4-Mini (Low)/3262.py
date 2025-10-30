from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array so that we can consider removing the largest elements first
        nums.sort()
        
        # Compute the total sum of all sides initially
        total = sum(nums)
        
        # Try to form the polygon using the largest i+1 sides (i from n-1 down to 2)
        # We maintain `total` as the sum of the current candidate sides.
        # At each step, the current largest side is nums[i], and the sum of the rest is total - nums[i].
        for i in range(len(nums) - 1, 1, -1):
            largest = nums[i]
            rest_sum = total - largest
            # Polygon condition: sum of the other sides > largest side
            if rest_sum > largest:
                return total
            # If it fails, remove the current largest from total and try with fewer sides
            total = rest_sum
        
        # If no valid polygon found
        return -1