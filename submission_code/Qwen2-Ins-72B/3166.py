from collections import Counter
from math import ceil

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        max_freq = max(count.values())
        min_groups = max_freq
        
        # Iterate over possible group sizes
        for group_size in range(1, max_freq + 1):
            # Calculate the number of groups needed for the current group size
            groups = sum(ceil(freq / (group_size + 1)) for freq in count.values())
            min_groups = min(min_groups, groups)
            
        return min_groups