from typing import List
from collections import defaultdict

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        if total_distinct == 0:
            return 0
        
        def atMostK(k):
            count = defaultdict(int)
            left = 0
            res = 0
            cur_distinct = 0
            for right in range(len(nums)):
                x = nums[right]
                if count[x] == 0:
                    cur_distinct += 1
                count[x] += 1
                
                while cur_distinct > k:
                    left_x = nums[left]
                    count[left_x] -= 1
                    if count[left_x] == 0:
                        cur_distinct -= 1
                    left += 1
                res += right - left + 1
            return res
        
        return atMostK(total_distinct) - atMostK(total_distinct - 1)