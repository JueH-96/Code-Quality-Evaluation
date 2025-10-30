from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        for i in range(n):
            freq = {}
            distinct = 0
            for j in range(i, n):
                num = nums[j]
                if num not in freq:
                    freq[num] = 1
                    distinct += 1
                else:
                    freq[num] += 1
                total_sum += distinct * distinct
                
        return total_sum