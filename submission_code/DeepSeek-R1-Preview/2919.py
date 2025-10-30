class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        left = 1
        right = 10**7  # A sufficiently large upper bound
        ans = 0
        
        while left <= right:
            mid = (left + right) // 2
            total = 0
            required = mid * (mid + 1) // 2
            for num in usageLimits:
                total += min(num, mid)
                if total >= required:
                    break
            if total >= required:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans