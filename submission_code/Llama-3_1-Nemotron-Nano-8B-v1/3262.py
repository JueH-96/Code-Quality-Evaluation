from typing import List

class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * n
        prefix_sum[0] = nums[0]
        for i in range(1, n):
            prefix_sum[i] = prefix_sum[i-1] + nums[i]
        
        for k in range(n, 2, -1):
            if k < 3:
                continue
            if prefix_sum[k-2] > nums[k-1]:
                return prefix_sum[k-1]
        
        return -1