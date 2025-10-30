class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort(reverse=True)
        total_groups = 0
        current_group_size = 0
        
        for limit in usageLimits:
            current_group_size += 1
            if limit < current_group_size:
                current_group_size = limit
            
            total_groups += current_group_size
            if total_groups < current_group_size * (current_group_size + 1) // 2:
                break
        
        return current_group_size