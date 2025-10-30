from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)

        # Helper function to check if a subarray can be split
        def canSplit(subarray):
            total_sum = sum(subarray)
            if total_sum >= m or len(subarray) == 1:
                return True
            for i in range(1, len(subarray)):
                left_sum = sum(subarray[:i])
                right_sum = sum(subarray[i:])
                if left_sum >= m and right_sum >= m:
                    return True
            return False

        # Try to split the array step by step
        while len(nums) < n:
            found_split = False
            for i in range(1, len(nums)):
                left = nums[:i]
                right = nums[i:]
                if canSplit(left) and canSplit(right):
                    nums = right
                    found_split = True
                    break
            if not found_split:
                return False

        return True