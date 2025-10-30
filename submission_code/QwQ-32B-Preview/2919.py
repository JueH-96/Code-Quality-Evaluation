class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        total_sum = sum(usageLimits)
        n = len(usageLimits)
        
        # Binary search to find the maximum k
        low, high = 1, n
        max_k = 0
        while low <= high:
            mid = (low + high) // 2
            required_sum = mid * (mid + 1) // 2
            if required_sum <= total_sum and mid <= n:
                max_k = mid
                low = mid + 1
            else:
                high = mid - 1
        return max_k