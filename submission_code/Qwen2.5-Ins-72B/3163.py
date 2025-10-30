from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_count = len(set(subarray))
                total_sum += distinct_count ** 2
        
        return total_sum