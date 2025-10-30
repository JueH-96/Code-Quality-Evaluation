from typing import List

class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        usageLimits = sorted((limit, i) for i, limit in enumerate(usageLimits))
        groups = []
        for limit, _ in usageLimits:
            added = False
            for group in groups:
                if len(group) < limit:
                    group.append(limit)
                    added = True
                    break
            if not added:
                groups.append([limit])
        return len(groups)