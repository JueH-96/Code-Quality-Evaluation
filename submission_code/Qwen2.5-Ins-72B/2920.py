from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        last_seen = defaultdict(int)
        max_dist = defaultdict(int)
        
        for i, num in enumerate(nums):
            if num in last_seen:
                dist = (i - last_seen[num]) // 2
                max_dist[num] = max(max_dist[num], dist)
            last_seen[num] = i
        
        # Check the distance between the last and first occurrence of each number
        for num in last_seen:
            dist = (n - last_seen[num] + nums.index(num)) // 2
            max_dist[num] = max(max_dist[num], dist)
        
        return min(max_dist.values())