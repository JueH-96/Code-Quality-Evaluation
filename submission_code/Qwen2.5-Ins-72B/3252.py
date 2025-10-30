from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        def is_strictly_increasing(arr):
            return all(arr[i] < arr[i + 1] for i in range(len(arr) - 1))
        
        n = len(nums)
        count = 0
        
        for start in range(n):
            for end in range(start, n):
                subarray = nums[start:end + 1]
                remaining = nums[:start] + nums[end + 1:]
                if is_strictly_increasing(remaining):
                    count += 1
        
        return count