from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        # Sort the array to easily check conditions against the k smallest/largest values.
        nums.sort()

        ways = 0

        # Case 1: 0 students selected (k=0).
        # All students are not selected. For each student i, we need 0 < nums[i].
        # This is true if and only if the smallest element, nums[0], is greater than 0.
        if nums[0] > 0:
            ways += 1

        # Case 2: k students selected, where 1 <= k < n.
        # For a group of size k to be happy, we must select the k students
        # with the smallest nums values.
        # For selected students, their `nums` value must be less than k.
        # This means nums[k-1] < k.
        # For unselected students, their `nums` value must be greater than k.
        # This means nums[k] > k.
        # So we check for nums[k-1] < k < nums[k].
        for k in range(1, n):
            if nums[k-1] < k and k < nums[k]:
                ways += 1
        
        # Case 3: n students selected (k=n).
        # All students are selected. For each student i, we need n > nums[i].
        # This is always true due to the problem constraint 0 <= nums[i] < n.
        ways += 1

        return ways