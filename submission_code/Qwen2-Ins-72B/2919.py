from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        total_elements = 0
        groups = 0
        for i, limit in enumerate(usageLimits):
            total_elements += limit
            # Check if we can form a new group with the current total elements
            if total_elements >= (groups + 1) * (groups + 2) // 2:
                groups += 1
        return groups