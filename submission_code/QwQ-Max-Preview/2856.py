from typing import List
from collections import defaultdict

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        n = len(nums)
        if total_distinct == 0:
            return 0
        max_possible = n * (n + 1) // 2
        
        def at_most(m):
            count = defaultdict(int)
            left = 0
            res = 0
            distinct = 0
            for right in range(n):
                x = nums[right]
                if count[x] == 0:
                    distinct += 1
                count[x] += 1
                while distinct > m:
                    y = nums[left]
                    count[y] -= 1
                    if count[y] == 0:
                        distinct -= 1
                    left += 1
                res += right - left + 1
            return res
        
        m = total_distinct - 1
        subarrays_at_most_m = at_most(m)
        return max_possible - subarrays_at_most_m