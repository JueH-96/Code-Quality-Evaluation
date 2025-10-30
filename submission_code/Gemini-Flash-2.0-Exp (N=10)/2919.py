import heapq

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        groups = 0
        current_group_size = 1
        remaining_elements = 0
        
        for limit in usageLimits:
            remaining_elements += limit
            if remaining_elements >= current_group_size:
                remaining_elements -= current_group_size
                groups += 1
                current_group_size += 1
        
        return groups