class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        n = len(usageLimits)
        total = sum(usageLimits)
        
        left, right = 1, n + min(total, n)
        ans = 0
        
        def is_possible(k):
            required = k * (k + 1) // 2
            total_cap = 0
            for u in reversed(usageLimits):
                if u >= k:
                    total_cap += k
                else:
                    total_cap += u
                if total_cap >= required:
                    return True
            return total_cap >= required
        
        while left <= right:
            mid = (left + right) // 2
            if is_possible(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans