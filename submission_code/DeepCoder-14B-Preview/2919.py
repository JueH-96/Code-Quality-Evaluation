class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        total = sum(usageLimits)
        n = len(usageLimits)
        low = 1
        high = n
        max_k = 0
        while low <= high:
            mid = (low + high) // 2
            required = mid * (mid + 1) // 2
            if required <= total:
                max_k = mid
                low = mid + 1
            else:
                high = mid - 1
        return max_k