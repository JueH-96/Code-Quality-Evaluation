from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()  # Sort the array in ascending order
        n = len(nums)
        sum_so_far = 0
        ans = -1
        for i in range(n):
            sum_so_far += nums[i]
            if i >= 2:  # We have at least 3 elements now
                if sum_so_far - nums[i] > nums[i]:  # Sum of the first i elements minus the largest > largest
                    ans = sum_so_far  # Update ans; it will be overwritten if a larger valid perimeter is found later
        return ans