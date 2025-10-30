class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        def canCreateKGroups(k):
            total_capacity = sum(min(limit, k) for limit in usageLimits)
            total_needed = k * (k + 1) // 2
            return total_capacity >= total_needed
        
        left, right = 1, len(usageLimits)
        while left < right:
            mid = (left + right + 1) // 2
            if canCreateKGroups(mid):
                left = mid
            else:
                right = mid - 1
        
        return left