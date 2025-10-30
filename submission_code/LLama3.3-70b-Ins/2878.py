from typing import List

class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        # Calculate the prefix sum of the array
        prefix_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Check if the prefix sum at each index is a multiple of k
        for i in range(k, len(nums) + 1):
            if prefix_sum[i] % k != prefix_sum[i - k] % k:
                return False
        
        # Check if the prefix sum at the end is 0
        return prefix_sum[-1] % k == 0