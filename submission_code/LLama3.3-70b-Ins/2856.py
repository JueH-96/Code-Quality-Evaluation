from typing import List

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        total_distinct = len(set(nums))
        count = 0
        
        for i in range(n):
            distinct = set()
            for j in range(i, n):
                distinct.add(nums[j])
                if len(distinct) == total_distinct:
                    count += 1
                    
        return count