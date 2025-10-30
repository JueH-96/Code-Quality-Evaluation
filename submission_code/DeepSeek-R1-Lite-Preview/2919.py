class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        n = len(usageLimits)
        usageLimits.sort()
        
        # Compute prefix sums from the end
        prefix_sum = [0] * (n + 1)
        for k in range(1, n + 1):
            prefix_sum[k] = prefix_sum[k - 1] + usageLimits[n - k]
        
        # Binary search for the maximum k
        low, high = 1, n
        result = 0
        while low <= high:
            mid = (low + high) // 2
            required = mid * (mid + 1) // 2
            if prefix_sum[mid] >= required:
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        return result