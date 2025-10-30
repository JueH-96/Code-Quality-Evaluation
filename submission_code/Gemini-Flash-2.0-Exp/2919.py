class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        total_sum = 0
        group_count = 0
        for limit in usageLimits:
            total_sum += limit
            if total_sum >= (group_count + 1) * (group_count + 2) // 2:
                group_count += 1
        return group_count