from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_increasing(arr):
            for i in range(len(arr) - 1):
                if arr[i] >= arr[i + 1]:
                    return False
            return True
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i, n):
                new_nums = nums[:i] + nums[j+1:]
                if is_increasing(new_nums):
                    count += 1
        return count