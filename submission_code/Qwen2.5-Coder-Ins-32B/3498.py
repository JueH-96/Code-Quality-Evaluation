from typing import List
from collections import defaultdict

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        group_count = [0] * k
        group_freq = [defaultdict(int) for _ in range(k)]
        
        for i in range(n):
            group = i % k
            group_count[group] += 1
            group_freq[group][nums[i]] += 1
        
        min_changes = [float('inf')] * (k + 1)
        min_changes[0] = 0
        
        for group in range(k):
            current_min = min(min_changes)
            new_min_changes = [current_min + group_count[group]] * (k + 1)
            
            for num, freq in group_freq[group].items():
                for diff in range(k):
                    new_min_changes[(num - diff) % k] = min(new_min_changes[(num - diff) % k], min_changes[diff] + group_count[group] - freq)
            
            min_changes = new_min_changes
        
        return min_changes[0]