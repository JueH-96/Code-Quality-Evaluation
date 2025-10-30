from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[:i] + nums[j+1:]
                if all(subarray[k] < subarray[k+1] for k in range(len(subarray)-1)):
                    count += 1
        
        return count