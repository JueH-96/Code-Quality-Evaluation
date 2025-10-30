class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        sorted_usage = sorted(usageLimits)
        max_k = 0
        for j in range(1, len(sorted_usage) + 1):
            if sorted_usage[j - 1] >= j:
                max_k = j
            else:
                break
        return max_k